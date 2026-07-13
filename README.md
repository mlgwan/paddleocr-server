# paddleocr-server
A minimal setup to host [PaddleOCR](https://github.com/PADDLEPADDLE/PADDLEOCR) as an API on a server.

The /ocr endpoint allows single file upload and can be used to quickly test on a specific machine. The settings are tuned to my specific hardware and are subject to change.


An endpoint for testing a single image is provided under [https://projects.ccl97.com/paddleocr/ocr](https://projects.ccl97.com/paddleocr/ocr) (POST)
and can be tested in a browser under [https://projects.ccl97.com/paddleocr/docs](https://projects.ccl97.com/paddleocr/docs)
# Example

This image:


<img width="1650" height="754" alt="image" src="https://github.com/user-attachments/assets/e65c680f-e855-4c1e-ad45-64a8422e030e" />


produced the following JSON output:

```
{
   "processing_time": 6.281,
   "predictions": 
   {
      "res": 
      {
         "input_path": null,
         "page_index": null,
         "model_settings": 
         {
            "use_doc_preprocessor": false,
            "use_textline_orientation": false
         },
         "dt_polys": 
         [
            /* omitted for brevity */
         ],
         "text_det_params": 
         {
            "limit_side_len": 64,
            "limit_type": "min",
            "thresh": 0.3,
            "max_side_limit": 4000,
            "box_thresh": 0.6,
            "unclip_ratio": 1.5
         },
         "text_type": "general",
         "textline_orientation_angles": 
         [
            /* omitted for brevity */
         ],
         "text_rec_score_thresh": 0,
         "return_word_box": false,
         "rec_texts": 
         [
            "",
            "PAPRIKA ROT",
            "EUR",
            "0,342 kg x",
            "1,71B",
            "4,99 EUR/kg",
            "RISPENTOMATE",
            "0,19B",
            "0,146kgx",
            "1,29 EUR/kg",
            "MISCHSALAT",
            "ROHK.",
            "0,",
            "79",
            "SALATGURKE",
            "79",
            "PIZZA VEGETALE",
            "3,49",
            "THUNFISCHFI.",
            "1,29",
            "B",
            "WAS"
         ],
         "rec_scores": 
         [
            0,
            0.982167661190033,
            0.9943094253540039,
            0.8664593696594238,
            0.9277381896972656,
            0.9375177621841431,
            0.99690842628479,
            0.8933364152908325,
            0.8923853635787964,
            0.9204270839691162,
            0.997593879699707,
            0.97517329454422,
            0.7774866223335266,
            0.9889017343521118,
            0.9970035552978516,
            0.991814374923706,
            0.9683261513710022,
            0.8831555843353271,
            0.9892482161521912,
            0.9167994260787964,
            0.9251604080200195,
            0.9974985718727112
         ],
         "rec_polys": 
         [
            /* omitted for brevity */
         ],
         "rec_boxes": 
         [
            /* omitted for brevity */
         ]
      }
   }
}
```
