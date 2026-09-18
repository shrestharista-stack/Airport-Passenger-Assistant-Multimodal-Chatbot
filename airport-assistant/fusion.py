# rule-based fusion (assignment section 4)
# not an MLP. CLIP and MiniLM scores are combined with simple rules.

from text_search import TextSearcher
from vision_search import ImageSearcher
from audio_pipeline import WhisperTranscriber


MIN_SCORE = 0.30


class AirportAssistant:
    def __init__(self, data_dir, device=None):
        self.data_dir = data_dir
        print("Building text index...")
        self.text_searcher = TextSearcher(data_dir)
        print("Building image index...")
        self.image_searcher = ImageSearcher(data_dir, device=device)
        print("Loading Whisper tiny...")
        self.whisper = WhisperTranscriber()

        self.info_desk = self.text_searcher.kb[0]
        for rec in self.text_searcher.kb:
            if rec["id"] == "info_t1":
                self.info_desk = rec
                break

    def refuse(self, mode, score, transcript=None):
        if score is None:
            score = 0.0
        return {
            "mode": mode,
            "matched": False,
            "score": round(float(score), 3),
            "name": None,
            "category": None,
            "description": None,
            "direction_text": None,
            "opening_hours": None,
            "accessibility": None,
            "assistance_contact": self.info_desk["assistance_contact"],
            "transcript": transcript,
            "message": "I am not sure from that input. Please ask staff at the information desk. This helper is not live airport data.",
        }

    def from_record(self, rec, mode, score, transcript=None):
        return {
            "mode": mode,
            "matched": True,
            "score": round(float(score), 3),
            "name": rec["name"],
            "category": rec["category"],
            "description": rec["description"],
            "direction_text": rec["direction_text"],
            "opening_hours": rec["opening_hours"],
            "accessibility": rec["accessibility"],
            "assistance_contact": rec["assistance_contact"],
            "transcript": transcript,
            "message": None,
        }

    def answer(self, text=None, image_path=None, audio_path=None):
        transcript = None
        if audio_path is not None:
            transcript = self.whisper.transcribe(audio_path)
            if text is None or str(text).strip() == "":
                text = transcript
            else:
                text = str(text) + " " + transcript

        has_text = text is not None and str(text).strip() != ""
        has_image = image_path is not None

        if (not has_text) and (not has_image):
            return self.refuse("none", 0.0)

        if has_image and has_text:
            if audio_path is not None:
                mode = "image+voice"
            else:
                mode = "image+text"
            image_hit = self.image_searcher.search_image(image_path)
            rec, text_score, entities, _ = self.text_searcher.search_text(
                text, category=image_hit["category"]
            )
            score = 0.5 * image_hit["score"] + 0.5 * text_score
            if score < MIN_SCORE:
                return self.refuse(mode, score, transcript)
            return self.from_record(rec, mode, score, transcript)

        if has_image:
            image_hit = self.image_searcher.search_image(image_path)
            rec = self.image_searcher.pick_kb_record(
                self.text_searcher.kb,
                image_hit["category"],
                image_hit["embedding"],
            )
            if rec is None or image_hit["score"] < MIN_SCORE:
                return self.refuse("image", image_hit["score"])
            return self.from_record(rec, "image", image_hit["score"])

        if audio_path is not None:
            mode = "voice"
        else:
            mode = "text"
        rec, score, entities, _ = self.text_searcher.search_text(text)
        if score < MIN_SCORE:
            return self.refuse(mode, score, transcript)
        return self.from_record(rec, mode, score, transcript)


def answer(assistant, text=None, image_path=None, audio_path=None):
    return assistant.answer(text=text, image_path=image_path, audio_path=audio_path)
