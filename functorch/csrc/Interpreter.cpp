#include <functorch/csrc/Interpreter.h>
#include <functorch/csrc/BatchedTensorImpl.h>
#include <functorch/csrc/TensorWrapper.h>
#include <functorch/csrc/VmapInterpreter.h>
#include <functorch/csrc/FunctionalizeInterpreter.h>
#include <functorch/csrc/ADInterpreters.h>

namespace at { namespace functorch {

static DispatchKeySet get_all_dynlayer_keyset() {
  // NB: FULL_AFTER does not include the dispatch key

  // "all dispatch keys between DynamicLayer{Front, Back}Mode, inclusive"
  auto result =
    DispatchKeySet(DispatchKeySet::FULL_AFTER, kDynamicLayerFrontModeKey) -
    DispatchKeySet(DispatchKeySet::FULL_AFTER, kDynamicLayerBackModeKey);
  result = result | DispatchKeySet({kDynamicLayerFrontModeKey});

  // Hack: don't handle the autocast dispatch keys. Their interaction with functorch
  // is weird.
  result = result - autocast_dispatch_keyset;

  // Hack: don't handle kVmapModeKey. We need a better way of modeling this.
  // In e.g. grad(vmap(f)), kVmapModeKey makes it so that all random operations,
  // even after we are done handling the vmap layer, error out.
  result = result.remove(kVmapModeKey);

  return result;
}

// TODO: This should be constexpr, but there are some methods
// of DispatchKeySet that haven't been marked constexpr yet.
static DispatchKeySet all_dynlayer_keyset = get_all_dynlayer_keyset();

static DispatchKeySet keysForEnteringDynamicLayer(TransformType key) {
  if (key == TransformType::Vmap) {
    // NB: Does not include kVmapModeKey. We may modulate the key when
    // constructing the DynamicLayer, but we don't control it when entering/exiting
    // the DynamicLayer.
    return DispatchKeySet({kBatchedKey});
  } else if (key == TransformType::Grad || key == TransformType::Jvp) {
    return autograd_dispatch_keyset.add(DispatchKey::ADInplaceOrView);
  } else if (key == TransformType::Functionalize) {
    return DispatchKeySet(DispatchKey::Functionalize);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported key: ", key);
  }
}

DispatchKeySet keysToExcludeWhenEnteringDynamicLayer(TransformType key) {
  DispatchKeySet exclude = all_dynlayer_keyset;
  exclude = exclude.remove(kDynamicLayerBackModeKey);
  exclude = exclude - keysForEnteringDynamicLayer(key);
  return exclude;
}

void setup_dispatch_key_tls(DispatchKeySet exclude, DispatchKeySet include) {
  auto local_keyset = c10::impl::tls_local_dispatch_key_set();
  local_keyset.excluded_ = local_keyset.excluded_ | exclude;
  local_keyset.included_ = local_keyset.included_ | include;
  c10::impl::_force_tls_local_dispatch_key_set(local_keyset);
}

std::ostream& operator<<(std::ostream& os, const TransformType& t) {
  switch (t) {
    case TransformType::Torch:
      os << "Torch";
      break;
    case TransformType::Vmap:
      os << "Vmap";
      break;
    case TransformType::Grad:
      os << "Grad";
      break;
    case TransformType::Jvp:
      os << "Jvp";
      break;
    case TransformType::Functionalize:
      os << "Functionalize";
      break;
    default:
      TORCH_INTERNAL_ASSERT(false);
  }
  return os;
}

void sanityCheckStack(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  auto num_args = op.schema().arguments().size();
  foreachTensorInplace(*stack, stack->size() - num_args, stack->size(),
      [](const Tensor& tensor) {

        auto* wrapper = maybeGetTensorWrapper(tensor);
        TORCH_INTERNAL_ASSERT(wrapper == nullptr);
        auto* batched = maybeGetBatchedImpl(tensor);
        TORCH_INTERNAL_ASSERT(batched == nullptr);
        return tensor;
      });
}

#define INTERPRETER_DISPATCH(type, method) \
  switch (key()) { \
    case TransformType::Vmap: \
      return VmapInterpreterPtr(this). method; \
    case TransformType::Grad: \
      return GradInterpreterPtr(this). method; \
    case TransformType::Jvp: \
      return JvpInterpreterPtr(this). method; \
    case TransformType::Functionalize: \
      return FunctionalizeInterpreterPtr(this). method; \
    default: \
      TORCH_INTERNAL_ASSERT(false, "Unrecognized transform"); \
  }

void Interpreter::process(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  INTERPRETER_DISPATCH(key_, SINGLE_ARG(processImpl(op, stack)));
}

void Interpreter::sendToNextInterpreter(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  INTERPRETER_DISPATCH(key_, SINGLE_ARG(sendToNextInterpreterImpl(op, stack)));
}

}}
