from rectorch.data import NCRDataProcessing
from rectorch.utils import get_data_cfg
from rectorch import env, set_seed
from rectorch.models.nn.ncr import NCR

if __name__ == '__main__':
    env.init(device="cuda:0")
    set_seed(2022)
    data_cfg = get_data_cfg('ml100k_ncr')
    processor = NCRDataProcessing(data_cfg)
    dataset = processor.process_and_split()

    opt_conf = {
                "name" : "adam",
                "lr" : 0.001,
                "weight_decay" : 1e-4
                }

    model = NCR(dataset.n_users, dataset.n_items, opt_conf=opt_conf)

    # TODO dare la possibilita di passare il sampler direttamente

    model.train(dataset, valid_metric='ndcg@5', early_stop=0)

    # model = NCR.load_model("./saved_models/best_ncr_model.json", 'cpu')

    model.test(dataset)



