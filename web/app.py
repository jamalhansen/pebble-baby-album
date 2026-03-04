"""Flask web viewer for pebble journal. Read-only. No network requests."""
from datetime import date
from pathlib import Path

from flask import Flask, render_template, request, abort

from pebble.config import Config
from pebble.models import MilestoneTag, MOOD_EMOJI
from pebble.storage import iter_entries, load_entry, search_entries

PAGE_SIZE = 10


def create_app(config: Config) -> Flask:
    template_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"
    app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))
    app.config["JOURNAL_CONFIG"] = config

    @app.context_processor
    def inject_globals():
        return {
            "baby_name": config.baby.name,
            "mood_emoji": MOOD_EMOJI,
            "today": date.today().isoformat(),
        }

    @app.route("/")
    def index():
        page = int(request.args.get("page", 1))
        all_entries = list(iter_entries(config.storage.journal_dir))
        total = len(all_entries)
        start = (page - 1) * PAGE_SIZE
        entries = all_entries[start : start + PAGE_SIZE]
        has_next = start + PAGE_SIZE < total
        has_prev = page > 1
        return render_template(
            "index.html",
            entries=entries,
            page=page,
            has_next=has_next,
            has_prev=has_prev,
        )

    @app.route("/entry/<date_str>")
    def entry(date_str: str):
        try:
            entry_date = date.fromisoformat(date_str)
        except ValueError:
            abort(404)
        e = load_entry(entry_date, config.storage.journal_dir)
        if e is None:
            abort(404)
        return render_template("entry.html", entry=e)

    @app.route("/milestones")
    def milestones():
        entries = search_entries(config.storage.journal_dir, tag=MilestoneTag.FIRST)
        return render_template("milestones.html", entries=entries)

    @app.route("/search")
    def search():
        q = request.args.get("q", "").strip() or None
        tag_str = request.args.get("tag", "").strip() or None
        tag = None
        if tag_str:
            try:
                tag = MilestoneTag(tag_str)
            except ValueError:
                pass
        results = search_entries(config.storage.journal_dir, query=q, tag=tag)
        return render_template("search.html", results=results, q=q or "", tag=tag_str or "")

    return app
