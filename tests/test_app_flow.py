from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_demo_dataset_to_analysis_answer_flow():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "AI Data Analyst"

    next(button for button in app.button if button.label == "Load sample sales").click().run()
    assert not app.exception
    assert app.session_state["dataset_name"] == "sample_sales.csv"
    assert app.session_state["dataset_profile"]["rows"] == 12

    next(
        button for button in app.button
        if button.label == "Which product generated the highest sales?"
    ).click().run()
    assert app.text_area[0].value == "Which product generated the highest sales?"

    next(button for button in app.button if button.label == "Analyze").click().run()
    assert not app.exception
    analysis = app.session_state["analysis"]
    assert analysis["error"] is None
    assert analysis["sql"].startswith("SELECT")
    assert "Crest 27-inch Monitor" in analysis["answer"]
    assert "2560" in analysis["answer"]
    assert analysis["frame"].iloc[0]["Product"] == "Crest 27-inch Monitor"
    assert "Crest 27-inch Monitor" in analysis["key_insight"]
    assert "2560" in analysis["key_insight"]