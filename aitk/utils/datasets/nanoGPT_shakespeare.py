import os

def get_dataset(name):

    if name == "shakespeare texts":
        # URL of the raw file on GitHub
        url = 'https://raw.githubusercontent.com/ArtificialIntelligenceToolkit/datasets/master/shakespeare_all_texts/input.txt'

        # Download the file
        os.system(f'wget {url} -O input.txt')

        path = 'input.txt'

        return path
    
    elif name == "saved model":
        # URL of the raw file on GitHub
        url = 'https://raw.githubusercontent.com/ArtificialIntelligenceToolkit/datasets/master/shakespeare_all_texts/saved_nanoGPT.pt'

        os.system(f'wget {url} -O saved_model.pt')

        path = 'saved_model.pt'

        return path
    
    else:

        print("Wrong filename! Please try again.")

        return None