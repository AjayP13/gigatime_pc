import click

from commands.process_patient import process_patient
from commands.view_image import view_image


@click.group()
def cli() -> None:
    """GigaTIME CLI."""


cli.add_command(process_patient)
cli.add_command(view_image)


if __name__ == "__main__":
    cli()
