import torch, logging, argparse, time, os, sys
from collections.abc import Sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import DataReader, Answer, Misconception
from src.dataset import AnswerDataset
from src.token import FunnyTokenizer
from src.model import RNNEncoder, CosScorer, BiEncoder


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception as e:
            print(e)
            self.handleError(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--data_dir", type=str, default="./dataset")
    parser.add_argument("--save_path", type=str, default="./model")
    parser.add_argument("--dev", type=str, default="cuda")

    args = parser.parse_args()

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # handler = TqdmLoggingHandler()
    info_handler = logging.StreamHandler(sys.stdout)
    debug_handler = logging.FileHandler("debug.log")
    info_handler.setLevel(logging.INFO)
    debug_handler.setLevel(logging.DEBUG)
    info_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    logger.addHandler(info_handler)
    logger.addHandler(debug_handler)

    reader = DataReader(args.data_dir)
    train_data, test_data = reader.read()
    logger.info(f"train data: {len(train_data)}")
    logger.info(f"test data: {len(test_data)}")

    tok = FunnyTokenizer(max_len=128)

    # construct corpus
    logger.info("construct corpus...")
    for question in tqdm(train_data):
        tok(question.text)
        tok(question.subject.name)
        for answer in question.answers:
            tok(answer.text)
            if answer.misconception:
                tok(answer.misconception.name)
    logger.info("corpus size: %d", len(tok))

    def collate_fn(batch: Sequence[tuple[Answer, Misconception, int]]):
        x = []
        y = []
        labels = []
        for answer, misconception, label in batch:
            x.append(
                tok(
                    f"{answer.question.subject.name} {tok.SEP} {answer.question.text} {tok.SEP} {answer.text}"
                )
            )
            y.append(tok(misconception.name))
            labels.append(label)
        x = torch.stack(x)
        y = torch.stack(y)
        labels = torch.tensor(labels, dtype=torch.float32)
        return x, y, labels

    dataset = AnswerDataset(train_data)
    train_loader: Sequence[
        tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]
    ] = DataLoader(
        dataset=dataset.train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    encoder = RNNEncoder(dim=args.dim, max_len=128, voc_len=len(tok))
    scorer = CosScorer()
    model = BiEncoder(encoder, encoder, scorer)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = torch.nn.CrossEntropyLoss()

    logger.info("start training...")
    model_name = time.strftime("%Y%m%d%H%M%S")
    dev = torch.device(args.dev) if torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)
    model.train()
    for epoch in tqdm(range(args.epochs)):
        acc_loss = list[float]()
        for step, (x, y, label) in enumerate(tqdm(train_loader, leave=False)):
            x = x.to(dev)
            y = y.to(dev)
            label = label.to(dev)
            sim = model(x, y)
            loss = criterion(sim, label)
            acc_loss.append(loss.item())

            model.zero_grad()
            loss.backward()
            optimizer.step()

            logger.debug(f"step: {epoch}-{step}, loss: {loss.item()}")

        logger.info(f"epoch: {epoch}, loss: {sum(acc_loss)/len(acc_loss)}")
        torch.save(model.state_dict(), os.path.join(args.save_path, model_name))
