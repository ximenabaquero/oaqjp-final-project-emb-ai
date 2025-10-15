import unittest
from EmotionDetection import emotion_detector

class TestEmotionDetection(unittest.TestCase):
    def test_joy(self):
        r = emotion_detector("I'm glad this happened.")
        self.assertEqual(r["dominant_emotion"], "joy", msg=r)

    def test_anger(self):
        r = emotion_detector("I'm really angry about this.")
        self.assertEqual(r["dominant_emotion"], "anger", msg=r)

    def test_disgust(self):
        r = emotion_detector("I feel disgusted just hearing about this.")
        self.assertEqual(r["dominant_emotion"], "disgust", msg=r)

    def test_sadness(self):
        r = emotion_detector("I'm so sad about this.")
        self.assertEqual(r["dominant_emotion"], "sadness", msg=r)

    def test_fear(self):
        # 'terrified' ayuda a disparar miedo explícitamente
        r = emotion_detector("I am terrified this will happen.")
        self.assertEqual(r["dominant_emotion"], "fear", msg=r)

if __name__ == "__main__":
    unittest.main()
