import pandas as pd
import numpy as np
import math
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel

class DataProcessor:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path

    def process_data(self):
        data = pd.read_csv(self.data_path, sep=';', dtype=object)
        data = data.reset_index(drop=True)
        data['text'] = data['Title'] + ' ' + data['Abstract']
        return data


class EmbeddingProcessor:
    def __init__(self, tokenizer, model, pca):
        self.tokenizer = tokenizer
        self.model = model
        self.pca = pca

    def get_embeddings(self, data):
        embedding = []
        pubmed_id = []
        function_r = []
        lcr = []
        hard = []
        category = []
        try:
            print('Calculating embeddings in progress ...')
            for i in range(len(data)):
                inputs = self.tokenizer(str(data['text'][i]), return_tensors="pt")
                key, val = next(iter(inputs.items()))
                pubmed_id.append(data['PMID'][i])
                function_r.append(data['Function'][i])
                lcr.append(data['LCR'][i])
                hard.append(data['Hard'].iloc[i])
                category.append(data['Category'].iloc[i])
                '''if the length of the text is greater than the max_seq_length then the text is split 
                and the calculated vectors in each group are combined to get the embedding describing the publication'''
                if len(val[0]) > 512:
                    begin = 0
                    end = 512
                    dataframe_tmp = pd.DataFrame()
                    for l in range(math.ceil(len(str(data['text'][i])) / 512)):
                        input_l = self.tokenizer(str(data['text'][i][begin:end]), return_tensors="pt")
                        output_l = self.model(**input_l)
                        test_l = pd.DataFrame(output_l['last_hidden_state'][0].tolist())
                        begin = begin + 512
                        end = end + 512
                        dataframe_tmp = pd.concat([dataframe_tmp, test_l])
                    frames = [dataframe_tmp]
                    result = pd.concat(frames)
                    embedding.append(np.transpose(self.pca.fit_transform(np.transpose(result)))[0])
                else:
                    inputs = self.tokenizer(str(data['text'][i]), return_tensors="pt")
                    outputs = self.model(**inputs)
                    test = pd.DataFrame(outputs['last_hidden_state'][0].tolist())
                    embedding.append(np.transpose(self.pca.fit_transform(np.transpose(test)))[0])
        except UnicodeDecodeError:
            pass

        dataframe = pd.DataFrame(embedding)
        dataframe['PMID'] = pubmed_id
        dataframe['Function'] = function_r
        dataframe['Lcr'] = lcr
        dataframe['Hard'] = hard
        dataframe['Category'] = category
        return dataframe


def main():
    data_path = './sample_data/annotated_sample_data.csv'
    model_path = "google-bert/bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = AutoModel.from_pretrained(model_path)
    pca = PCA(n_components=1)

    data_processor = DataProcessor(data_path, model_path)
    data = data_processor.process_data()

    embedding_processor = EmbeddingProcessor(tokenizer, model, pca)
    embedding_data = embedding_processor.get_embeddings(data)

    embedding_data.to_csv('./output/embedding_data/sample_BERT_embedding.csv', sep=';', index=False)

if __name__ == "__main__":
    main()