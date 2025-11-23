import click

@click.command()
@click.argument("config-path", type=click.Path(exists=True, dir_okay=False))
def main(config_path):
    trainer.test(config.model, data)
