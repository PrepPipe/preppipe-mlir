// RUN: preppipe-mlir-opt %s | FileCheck %s

module {
  // CHECK: preppipe.image_asset
  preppipe.image_asset @image "background.png", {width = 1920, height = 1080, bbox = #preppipe.bbox<100, 50, 800, 600>} : !preppipe.image

  // CHECK: preppipe.audio_asset
  preppipe.audio_asset @audio "bg_music.mp3", {duration = 120, format = "mp3", sample_rate = 44100} : !preppipe.audio

  // CHECK: preppipe.asset_decl
  preppipe.asset_decl @ref_image  "background.png" {Test2=1} {Test=1} : !preppipe.image
  // CHECK: preppipe.asset_decl
  preppipe.asset_decl @ref_audio "bg_music.mp3" {} : !preppipe.audio
}

