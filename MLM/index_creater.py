import faiss
import hydra
from tqdm.auto import tqdm
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
from utils import MLMDataIteration
from pprint import pprint


class FAISS:
    def __init__(self, cfg):
        self.dim = cfg.faiss.dimensions
        self.index = faiss.IndexFlatL2(self.dim)
        self.vectors = {}
        self.counter = 0
        self.model_name = cfg.faiss.embedding_model_name
        self.sentence_encoder = SentenceTransformer(self.model_name)

    def add(self, text, idx, emb=None):
        if emb is None:
            text_emb = self.sentence_encoder.encode([text])
        else:
            text_emb = emb
        self.index.add(text_emb)
        self.vectors[self.counter] = (idx, text)
        self.counter += 1

    def search(self, emb, top_k=10):
        result = []
        distance, item_idx = self.index.search(emb, top_k)
        for dist, i in zip(distance[0], item_idx[0]):
            result.append((self.vectors[i][1], dist))
        return result


@hydra.main(config_path="/Users/ddmitriev/PycharmProjects/MIPTMasterThesis/config/", config_name="mlm_config")
def main(cfg: DictConfig):
    test_iter = MLMDataIteration(cfg)
    index = FAISS(cfg)
    for idx, i in tqdm(enumerate(test_iter), total=cfg.MLMData.n_samples):
        index.add(i, idx)

    test_emb = index.sentence_encoder.encode([i])
    pprint(index.search(test_emb), width=130)


if __name__ == '__main__':
    main()
