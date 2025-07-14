
#ifndef PREPPIPE_PREPPIPE_OPS
#define PREPPIPE_PREPPIPE_OPS

include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"

class PrepPipe_Op<string mnemonic, list<Trait> traits = []>
    : Op<PrepPipe_Dialect, mnemonic, traits> {
}

def MetadataOp : PrepPipe_Op<"metadata", []> {
  let summary = "Attach metadata to the IR";
  let description = [{
    This op is used to tag metadata such as error codes or user comments.
    该 Op 用于标记元数据，例如错误码或用户的注释。

    It carries two string attributes:
    - `kind`: A short identifier for the type of metadata, either an error code
       or message kind (e.g., "comment")
    - `message`: The full message string to display or propagate
    它携带两个字符串属性：
    - `kind`: 元数据类型的简短标识符，可以是错误码或消息类型（例如 "comment"）。
    - `message`: 要显示或传递的完整消息内容
  }];

  let arguments = (ins
    StrAttr:$kind,
    StrAttr:$message
  );

  let results = (outs);

  let assemblyFormat = [{
    `<` $kind `>` $message attr-dict
  }];
}


#endif // PREPPIPE_PREPPIPE_OPS
