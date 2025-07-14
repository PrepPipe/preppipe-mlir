#ifndef PREPPIPE_PREPPIPE_BASE_TD
#define PREPPIPE_PREPPIPE_BASE_TD

include "mlir/IR/OpBase.td"

def PrepPipe_Dialect : Dialect {
  let name = "preppipe";
  let cppNamespace = "::mlir::preppipe::PrepPipe";
  let summary = "Top-level dialect for PrepPipe's visual novel lowering pipeline";
  let description = [{
    Top-level dialect for PrepPipe Compiler's visual novel lowering pipeline.
    语涵编译器的视觉小说下沉管线的顶层方言。

    This dialect contains operations, attributes, etc. that are used throughout
    the PrepPipe lowering pipeline. They are generally not converted during the
    lowering process. Operations, etc. that are specific to one particular
    abstraction level are defined in the dialects for those levels.
    该方言包含在语涵编译器下沉管线中全流程使用的操作、属性等。它们通常在下沉过程中保持
    原样。特定于某一抽象级别的操作等在那些级别的方言中定义。
  }];
}

#endif // PREPPIPE_PREPPIPE_BASE_TD
