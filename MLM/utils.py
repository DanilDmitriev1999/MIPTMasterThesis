from datasets import load_dataset
from omegaconf import DictConfig
import hydra


class MLMDataIteration:
    def __init__(self, cfg: DictConfig):
        # TODO: prepare module for mc4 stream
        self.data_generator = load_dataset(cfg.MLMData.path_data,
                                           cfg.MLMData.language,
                                           split='train',
                                           streaming=True,)
        self.n_samples = cfg.MLMData.n_samples

    def __iter__(self):
        count = 0
        for sample in self.data_generator:
            text = sample['text']
            text_split = text.split('\n')
            for sub_text in text_split:
                if count < self.n_samples:
                    count += 1
                    yield sub_text
                else:
                    return


@hydra.main(config_path="/Users/ddmitriev/PycharmProjects/MIPTMasterThesis/config/", config_name="mlm_config")
def main(cfg: DictConfig):
    test_iter = MLMDataIteration(cfg)
    for idx, i in enumerate(test_iter):
        if idx == 10000:
            break
        print(i)


if __name__ == '__main__':
    main()
