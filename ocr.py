
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class OCR:
    """
    Transcribe the text of a comment
    """

    def __init__(self, thresh=0.9):
        """Constructor

        :param thresh: probably threshold which detects a comment (experimental), defaults to 0.9
        """
        self.thresh = thresh

        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

        # Beam search parameters (need this to get one single score (prob))
        # Adapted from https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.model.config.max_length = 10
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4
        self.max_length = 50

    def run(self, image):
        """Run transcription 

        :param image: input image
        :return: text, score
        """
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        return_dict = self.model.generate(pixel_values, max_length=self.max_length, output_scores=True, return_dict_in_generate=True)

        generated_ids, scores = return_dict['sequences'], return_dict['sequences_scores']
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text, torch.exp(scores).item()
    
    def run_test(self, image):
        """Get score of transcription (experimental, didn't really work)

        :param image: input image
        :return: score
        """
        _, prob = self.run(image)
        return prob >= self.thresh
    

if __name__ == "__main__":
    image = Image.open("test2.png").convert("RGB")

    ocr = OCR()
    text, prop = ocr.run(image)
    print(text, prop)