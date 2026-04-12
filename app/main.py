import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.rag.rag import ask_rag
from backend.vision.vision import detect_face

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_INDEX_DIR = BASE_DIR / "backend" / "rag" / "faiss_index"

db = FAISS.load_local(
    str(FAISS_INDEX_DIR),
    embedding_model,
    allow_dangerous_deserialization=True,
)


def main():
    print("CHALLENGERS AI Assistant Started")
    user_name = None
    chat_history = []

    while True:
        print("\nChoose Mode:")
        print("1. Chat")
        print("2. Camera (Face Recognition)")
        print("3. Exit")

        choice = input("Enter choice: ").strip()

        if choice == "1":
            query = input("\nAsk: ")

            if user_name:
                query = f"User name is {user_name}. {query}"

            chat_history.append(f"User: {query}")
            history_text = "\n".join(chat_history[-5:])

            answer = ask_rag(query, db, client, history_text)
            print("\nAnswer:\n", answer)
            chat_history.append(f"Assistant: {answer}")

        elif choice == "2":
            name = detect_face()

            if name and name != "Unknown":
                user_name = name
                print(f"\nDetected: {user_name}")
                chat_history.append(f"System: Detected user {user_name}")
            else:
                print("\nNo face detected.")

        elif choice == "3":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
