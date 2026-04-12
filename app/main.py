
import sys
import os
import cv2
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.rag.rag import ask_rag
from backend.vision.vision import detect_face

from google import genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv



# 🔹 Load environment variables
load_dotenv()

# 🔹 Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 🔹 Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 🔹 Load FAISS database
BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_INDEX_DIR = BASE_DIR / "backend" / "rag" / "faiss_index"

db = FAISS.load_local(
    str(FAISS_INDEX_DIR),
    embedding_model,
    allow_dangerous_deserialization=True
)


# 🔥 MAIN LOOP
def main():
    print("CHALLENGERS AI Assistant Started")

    while True:
        print("\nChoose Mode:")
        print("1. Chat")
        print("2. Camera (Face Recognition)")
        print("3. Exit")

        choice = input("Enter choice: ").strip()

        # 🔹 Chat Mode
        if choice == "1":
            query = input("\nAsk: ")
            answer = ask_rag(query, db, client)
            print("\nAnswer:\n", answer)

        # 🔹 Camera Mode
        elif choice == "2":
            name = detect_face()

            if name:
                print(f"\n👁️ Detected: {name}")

                query = f"{name} has entered the CHALLENGERS club. Greet them and assist."
                answer = ask_rag(query, db, client)

                print("\nAssistant:\n", answer)
            else:
                print("\nNo face detected.")

        # 🔹 Exit
        elif choice == "3":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Try again.")


# 🔹 Run program
if __name__ == "__main__":
    main()

