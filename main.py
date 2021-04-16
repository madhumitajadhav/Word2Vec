from preprocess import preprocess
from training import train_skipgram


def main():
    # Use a breakpoint in the code line below to debug your script.
    print("Loading Data...")
    with open('data/text8') as f:
        text = f.read()

    print("Preprocessing Data...")
    preprocessed = preprocess(text)

    print("Training Model...")
    model = train_skipgram(preprocessed)
    print("Model has been trained !!")


if __name__ == '__main__':
    main()
