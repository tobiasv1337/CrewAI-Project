from streamlit.testing.v1 import AppTest

def test_catalog_usage_summaries_do_not_claim_absence_of_degree_mapping():
    script = '''
import streamlit as st
from core.models import MosesModuleData, MosesDegreeUsage, MosesCatalogAssignment
from ui.details import _render_degree_usage_section
moses = MosesModuleData(number="40113",version=7,title="Hardware Security Lab",degree_usages=[
    MosesDegreeUsage(degree_name="Computer Engineering (M. Sc.)",study_regulations_count=1,usage_count=3,first_usage="WiSe 2024/25",last_usage="WiSe 2024/25"),
    MosesDegreeUsage(degree_name="Computer Science",semester_assignments={"WS 24/25":[MosesCatalogAssignment(raw_catalog="Hardware")]})
])
_render_degree_usage_section(moses, key="test")
'''
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert not app.expander
    assert not app.dataframe
    assert any("does not mean the module has no degree mapping" in caption.value for caption in app.caption)
    assert not any("No expanded semester" in caption.value for caption in app.caption)
