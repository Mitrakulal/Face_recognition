import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.rag.rag import ask_rag
from backend.vision.vision import detect_face

load_dotenv()



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
        print("\n--- MENU ---")
        print("1. Chat")
        print("2. Camera (Face Recognition)")
        print("3. Exit")

        choice = input("Enter choice: ").strip()

        if choice == "1":
            raw_query = input("\nAsk: ")

            if raw_query.lower() in ["exit", "quit"]:
                break

            query = raw_query
            if user_name:
                query = f"User name is {user_name}. {raw_query}"

            chat_history.append(f"User: {raw_query}")

            history_text = "\n".join(chat_history[-5:])

            try:
                print("\nAnswer:\n", end="")
                answer = ask_rag(
                    query,
                    db,
                    history_text,
                    on_token=lambda token: print(token, end="", flush=True),
                )
                print()

                chat_history.append(f"Assistant: {answer}")

            except Exception:
                print("\n⚠️ LLM not responding. Check Ollama.")

        elif choice == "2":
            name = detect_face()

            if name and name != "Unknown":
                user_name = name
                print(f"\nDetected: {user_name}")
                chat_history.append(f"User name is {user_name}")
            else:
                print("\nNo face detected.")

        elif choice == "3":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
