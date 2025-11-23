import click
from pytorch_lightning import Trainer
from utils.config_utils import load_config
from data.data_module import DocDataModule


@click.command()
@click.argument("config-path", type=click.Path(exists=True, dir_okay=False))
def main(config_path):
    config = load_config(config_path)
    data = DocDataModule(config)
    trainer = Trainer(
        max_epochs=config.epochs,
        devices="auto",
        accelerator="auto",
    )

    trainer.fit(config.model, data)


if __name__ == "__main__":
    main()
