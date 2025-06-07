
# Example (Python):
from tflite_support import metadata as _metadata
from tflite_support.metadata_writers import image_segmenter

writer = image_segmenter.MetadataWriter.create_for_inference(
        model_buffer=open("background_segmentation.tflite", "rb").read(),
            input_norm_mean=[127.5],
                input_norm_std=[127.5],
                    labels=["background", "person"],
                        output_type=image_segmenter.SegmentationOutputType.CATEGORY_MASK,
)

metadata_buf = writer.populate()
with open("background_segmentation_with_metadata.tflite", "wb") as f:
    f.write(metadata_buf)

)