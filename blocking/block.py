from pathlib import Path

from paddleocr import PPStructureV3

input_path = "blocking/block1.png"
output_dir = Path("blocking/output")
output_dir.mkdir(parents=True, exist_ok=True)

pipeline = PPStructureV3(
    engine="transformers",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_formula_recognition=False,
    wireless_table_structure_recognition_model_name="SLANeXt_wireless",
)

for result in pipeline.predict(input=input_path):
    result.print()
    result.save_to_json(save_path=str(output_dir))
    result.save_to_markdown(save_path=str(output_dir))
    result.save_to_html(save_path=str(output_dir))
    result.save_to_img(save_path=str(output_dir))