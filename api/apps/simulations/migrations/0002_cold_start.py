"""Cold-start simulations: a run may replay a dataset without a trained fit."""

import django.db.models.deletion
from django.db import migrations, models


def link_datasets(apps, schema_editor):
    Simulation = apps.get_model("simulations", "Simulation")
    for sim in Simulation.objects.select_related("fit").filter(dataset__isnull=True, fit__isnull=False):
        sim.dataset_id = sim.fit.dataset_id
        sim.framework = sim.fit.framework or {}
        sim.save(update_fields=["dataset", "framework"])


class Migration(migrations.Migration):
    dependencies = [
        ("datasets", "0001_initial"),
        ("fits", "0001_initial"),
        ("simulations", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="simulation",
            name="fit",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.PROTECT,
                related_name="simulations",
                to="fits.fit",
            ),
        ),
        migrations.AddField(
            model_name="simulation",
            name="dataset",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.PROTECT,
                related_name="simulations",
                to="datasets.dataset",
            ),
        ),
        migrations.AddField(
            model_name="simulation",
            name="framework",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.RunPython(link_datasets, migrations.RunPython.noop),
    ]
