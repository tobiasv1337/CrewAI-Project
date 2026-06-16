"""Tool functions exposed to TU Study Assistant agents."""

from crew.tools.moses_tools import (
    MOSES_MODULE_RESEARCH_TOOLS,
    MOSES_TOOLS,
    get_degree_area_modules,
    get_degree_program_structure,
    get_module_catalogs,
    get_module_details,
    search_degree_modules,
    search_degree_programs,
    search_modules,
)
from crew.tools.isis_tools import (
    CONFIRMATION_TOKEN,
    ISIS_READ_ONLY_TOOLS,
    ISIS_TOOLS,
    ISIS_WRITE_TOOLS,
    make_isis_read_only_tools,
)
from crew.tools.grademanager_tools import (
    GRADE_MANAGER_READ_TOOLS,
    GRADE_MANAGER_TOOLS,
    GRADE_MANAGER_WRITE_TOOLS,
    STUDY_ADVISOR_TOOLS,
    STUDY_PLAN_CONFIRMATION_TOKEN,
    add_module_to_study_plan,
    build_student_plan_context,
    check_module_against_study_plan,
    get_degree_requirement_details,
    get_study_plan_snapshot,
    list_study_plan_modules,
)
from crew.tools.proposal_tools import (
    ProposeCourseActionsTool,
    collect_course_proposals,
    current_course_proposals,
)
from crew.tools.degree_regulations_tools import (
    DEGREE_REGULATIONS_TOOLS,
    ExtractRegelstudienplanTableTool,
    ListDegreeRegulationPdfsTool,
    SearchDegreeRegulationPdfsTool,
)

COURSE_COMMITMENT_TOOLS = [
    ProposeCourseActionsTool(),
    *GRADE_MANAGER_WRITE_TOOLS,
    *ISIS_WRITE_TOOLS,
]

__all__ = [
    "CONFIRMATION_TOKEN",
    "COURSE_COMMITMENT_TOOLS",
    "DEGREE_REGULATIONS_TOOLS",
    "ISIS_READ_ONLY_TOOLS",
    "ISIS_TOOLS",
    "ISIS_WRITE_TOOLS",
    "GRADE_MANAGER_READ_TOOLS",
    "GRADE_MANAGER_TOOLS",
    "GRADE_MANAGER_WRITE_TOOLS",
    "MOSES_TOOLS",
    "MOSES_MODULE_RESEARCH_TOOLS",
    "STUDY_ADVISOR_TOOLS",
    "STUDY_PLAN_CONFIRMATION_TOKEN",
    "ExtractRegelstudienplanTableTool",
    "ListDegreeRegulationPdfsTool",
    "ProposeCourseActionsTool",
    "SearchDegreeRegulationPdfsTool",
    "collect_course_proposals",
    "current_course_proposals",
    "make_isis_read_only_tools",
    "build_student_plan_context",
    "search_modules",
    "get_module_details",
    "get_module_catalogs",
    "search_degree_programs",
    "get_degree_program_structure",
    "get_degree_area_modules",
    "search_degree_modules",
    "get_study_plan_snapshot",
    "list_study_plan_modules",
    "get_degree_requirement_details",
    "check_module_against_study_plan",
    "add_module_to_study_plan",
]
