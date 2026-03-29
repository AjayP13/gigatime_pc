import click

from commands.process_patient import process_patient
from commands.view_image import view_image
from commands.view_whole_slide import view_whole_slide


@click.group()
def cli() -> None:
    """GigaTIME CLI."""


cli.add_command(process_patient)
cli.add_command(view_image)
cli.add_command(view_whole_slide)


if __name__ == "__main__":
    cli()
