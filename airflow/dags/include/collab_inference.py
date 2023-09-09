import click

from collaborative.pipelines import inference


@click.command()
@click.option(
    "-u",
    "--userids",
    "user_ids",
    default=[],
    type=int,
    multiple=True,
    nargs=1,
    help="Set User ID for inference.",
)
@click.option(
    "-n",
    "--nrecommendations",
    "n_recommendations",
    default=-1,
    type=int,
    nargs=1,
    help="Set a number of movie recommendations.",
)
def main(n_recommendations, user_ids):

    inference.run(list(user_ids), n_recommendations)


if __name__ == "__main__":
    main()
