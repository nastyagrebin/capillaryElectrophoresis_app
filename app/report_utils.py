import panel as pn
from typing import Dict
import io

class FinalReportController:
    def __init__(self):
        self.report_pane = pn.pane.Markdown("No session parameters recorded yet.", sizing_mode="stretch_width")
        self.export_btn = pn.widgets.FileDownload(
            label="Download Session Report (.txt)", 
            filename="session_report.txt",
            button_type="primary",
            embed=False,
            auto=False,
            callback=self._generate_report_bytes,
            disabled=True
        )
        self.session_log_ref: Dict[str, dict] = {}
        
        self.section = pn.Column(
            pn.pane.Markdown("## Final Session Report"),
            pn.pane.Markdown("Review the exact parameters used during this pipeline session. Export this report to save alongside your data for your lab notebooks."),
            self.export_btn,
            pn.layout.Divider(),
            self.report_pane,
            sizing_mode="stretch_width"
        )

    def _format_report(self) -> str:
        if not self.session_log_ref:
            return "No session parameters recorded yet."
            
        lines = []
        lines.append("=" * 50)
        lines.append(" CEtools Session Report")
        lines.append("=" * 50)
        lines.append("")
        
        # General info first
        if "General" in self.session_log_ref:
            lines.append("--- General Information ---")
            for k, v in self.session_log_ref["General"].items():
                if isinstance(v, list):
                    lines.append(f"{k}:")
                    for item in v:
                        lines.append(f"  - {item}")
                else:
                    lines.append(f"{k}: {v}")
            lines.append("")
            
        # Pipeline steps
        step_order = [
            "Preprocessing: Despike",
            "Preprocessing: Smooth",
            "Preprocessing: Baseline",
            "Preprocessing: Normalization",
            "Alignment",
            "NMF",
            "Alpha Diversity"
        ]
        
        for step in step_order:
            if step in self.session_log_ref:
                lines.append(f"--- {step} ---")
                for k, v in self.session_log_ref[step].items():
                    lines.append(f"{k}: {v}")
                lines.append("")
                
        # Any other steps
        for step, data in self.session_log_ref.items():
            if step not in step_order and step != "General":
                lines.append(f"--- {step} ---")
                for k, v in data.items():
                    lines.append(f"{k}: {v}")
                lines.append("")
                
        return "\n".join(lines)

    def update_preview(self, current_log: Dict[str, dict]):
        self.session_log_ref = current_log
        report_text = self._format_report()
        
        # Convert to Markdown for the UI
        md_text = f"```text\n{report_text}\n```"
        self.report_pane.object = md_text
        
        if current_log:
            self.export_btn.disabled = False
        else:
            self.export_btn.disabled = True

    def _generate_report_bytes(self) -> io.BytesIO:
        report_text = self._format_report()
        return io.BytesIO(report_text.encode('utf-8'))

def build_report_section() -> tuple[pn.Column, FinalReportController]:
    ctrl = FinalReportController()
    return ctrl.section, ctrl
