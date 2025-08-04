
#ifndef PREPPIPE_PREPPIPE_OPS
#define PREPPIPE_PREPPIPE_OPS

include "preppipe-mlir/Dialect/PrepPipe/IR/PrepPipeTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"

def MetadataOp : PrepPipe_Op<"metadata", []> {
  let summary = "Attach metadata to the IR";
  let description = [{
    This op is used to tag metadata such as error codes or user comments.

    It carries two string attributes:
    - `kind`: A short identifier for the type of metadata, either an error code
       or message kind (e.g., "comment")
    - `message`: The full message string to display or propagate
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

// ============================================================================
// Asset Declaration Operations
// ============================================================================

class AssetDeclarationOp<string mnemonic, list<Trait> traits = []> : PrepPipe_Op<mnemonic, !listconcat(traits, [Symbol])> {}

def ImageAssetOp : AssetDeclarationOp<"image_asset", []> {
  let summary = "Declare an image asset";
  let description = [{
    This op declares an image asset. It must be an existing file (embedded or standalone).
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name, // codename (identifier)
    StrAttr:$path, // file path
    DictionaryAttr:$imageattrs // additional attributes for the image (e.g., size, bounding box, etc.)
  );

  let results = (outs
    ImageType:$result
  );

  let assemblyFormat = "$sym_name $path `,` $imageattrs attr-dict `:` type($result)";
}

def AudioAssetOp : AssetDeclarationOp<"audio_asset", []> {
  let summary = "Declare an audio asset";
  let description = [{
    This op declares an audio asset. It must be an existing file (embedded or standalone).
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$path,
    DictionaryAttr:$audioattrs
  );

  let results = (outs
    AudioType:$result
  );

  let assemblyFormat = "$sym_name $path `,` $audioattrs attr-dict `:` type($result)";
}

def AssetDeclOp : AssetDeclarationOp<"asset_decl", []> {
  let summary = "Declare an existing asset";
  let description = [{
    This op declares an asset that we expect to exist but is currently not accessible.
    It can be any asset type (image, audio, etc.)
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$path,
    DictionaryAttr:$refattrs
  );

  let results = (outs
    AnyTypeOf<[ImageType, AudioType]>:$result
  );

  let assemblyFormat = "$sym_name $path $refattrs attr-dict `:` type($result)";
}

#endif // PREPPIPE_PREPPIPE_OPS
