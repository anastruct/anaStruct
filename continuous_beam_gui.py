import datetime
import json
import os
from tkinter import filedialog, messagebox

import customtkinter as ctk
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

from anastruct import SystemElements

UNIT_FORCE_TO_KN = {
    "kN": 1.0, "N": 0.001, "tf": 9.80665, "kgf": 0.00980665,
    "lb": 0.004448222, "kip": 4.4482216
}
UNIT_LENGTH_TO_M = {
    "m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048
}

# ---------- ตั้งค่า Font ภาษาไทยสำหรับกราฟทั้งหมด ----------
plt.rcParams["font.family"] = "Tahoma"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ContinuousBeamAnalyzerPro(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Continuous Beam Analyzer PRO (v2.0) - Ultimate Report & UI Fixed")
        self.geometry("1500x950")

        self.ss = None
        self.span_elem_map = {}

        self.spans_data = [
            {
                "L": 5.0,
                "Material": "Concrete",
                "E_custom": 200.0,
                "Section": "Rect (bxh)",
                "b": 0.2,
                "h": 0.4,
                "I_custom": 0.001333,
            }
        ]
        self.supports_data = ["Pinned", "Roller"]
        self.loads_data = [
            {
                "type": "Uniform",
                "case": "DL",
                "span_index": 0,
                "val": -10.0,
                "x_offset": 0.0,
            }
        ]
        self.load_factors = {"DL": 1.4, "LL": 1.7}
        self.unit_force = "kN"
        self.unit_length = "m"
        self._load_settings()  # load saved unit preferences

        from datetime import datetime

        self.project_data = {
            "name": "Warehouse Project",
            "location": "Bangkok",
            "engineer": "Phornej",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "beam_name": "B1",
        }

        self.setup_ui()
        self.refresh_ui()

    SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "beam_settings.json")

    def _load_settings(self):
        try:
            if os.path.exists(self.SETTINGS_FILE):
                with open(self.SETTINGS_FILE, "r", encoding="utf-8") as f:
                    s = json.load(f)
                self.unit_force = s.get("unit_force", "kN")
                self.unit_length = s.get("unit_length", "m")
        except Exception:
            pass

    def _save_settings(self):
        try:
            with open(self.SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump({"unit_force": self.unit_force, "unit_length": self.unit_length}, f)
        except Exception:
            pass

    def _rescale_graph_labels(self, fig, diagram_type):
        """Convert anastruct graph text labels (in kN/kN·m) to user-selected units."""
        import re
        ff = UNIT_FORCE_TO_KN[self.unit_force]
        lf = UNIT_LENGTH_TO_M[self.unit_length]
        fu = self.unit_force
        lu = self.unit_length
        ax = fig.gca()
        for text in ax.texts:
            t = text.get_text().strip()
            if diagram_type == "structure":
                m = re.match(r"^(q|F|T)=(-?[\d.eE+\-]+)", t)
                if m:
                    prefix, raw = m.group(1), float(m.group(2))
                    if prefix == "q":
                        text.set_text(f"q={raw / ff * lf:.3g}")
                    elif prefix == "F":
                        text.set_text(f"F={raw / ff:.3g}")
                    elif prefix == "T":
                        text.set_text(f"T={raw / (ff * lf):.3g}")
            elif diagram_type == "reaction":
                m = re.match(r"^R=(-?[\d.eE+\-]+)", t)
                if m:
                    text.set_text(f"R={float(m.group(1)) / ff:.3f}")
            elif diagram_type in ("sfd", "bmd"):
                try:
                    val = float(t)
                    conv = val / ff if diagram_type == "sfd" else val / (ff * lf)
                    text.set_text(f"{conv:.2f}")
                except ValueError:
                    pass

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1, minsize=450)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.left_frame.grid_rowconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(self.left_frame)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)

        self.tab_proj = self.tabview.add("0. Project Info")
        self.tab_geo = self.tabview.add("1. Geometry & Section")
        self.tab_load = self.tabview.add("2. Loads & Combos")
        self.tab_res = self.tabview.add("3. Results (Data)")

        bottom_menu = ctk.CTkFrame(self.left_frame)
        bottom_menu.grid(row=1, column=0, sticky="ew", padx=10, pady=10)

        self.chk_structure = ctk.BooleanVar(value=True)
        self.chk_reaction = ctk.BooleanVar(value=True)
        self.chk_sfd = ctk.BooleanVar(value=True)
        self.chk_bmd = ctk.BooleanVar(value=True)
        self.chk_deflection = ctk.BooleanVar(value=True)

        ctk.CTkCheckBox(
            bottom_menu,
            text="Structure",
            variable=self.chk_structure,
            command=self.update_plots,
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkCheckBox(
            bottom_menu,
            text="Reaction",
            variable=self.chk_reaction,
            command=self.update_plots,
        ).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkCheckBox(
            bottom_menu, text="SFD", variable=self.chk_sfd, command=self.update_plots
        ).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkCheckBox(
            bottom_menu, text="BMD", variable=self.chk_bmd, command=self.update_plots
        ).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkCheckBox(
            bottom_menu,
            text="Deflection",
            variable=self.chk_deflection,
            command=self.update_plots,
        ).grid(row=2, column=0, padx=5, pady=5, sticky="w")

        self.switch_theme = ctk.CTkSwitch(
            bottom_menu, text="🌙 Dark Mode", command=self.toggle_theme
        )
        self.switch_theme.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.switch_theme.select()  # Default is dark

        self.btn_calc = ctk.CTkButton(
            bottom_menu,
            text="⚙️ อัปเดตโมเดล & วิเคราะห์",
            height=36,
            font=("Helvetica", 14, "bold"),
            command=self.calculate,
        )
        self.btn_calc.grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 5)
        )

        # Keyboard Shortcuts
        self.bind("<Control-Return>", lambda event: self.calculate())

        file_menu = ctk.CTkFrame(bottom_menu, fg_color="transparent")
        file_menu.grid(row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=(0, 5))

        ctk.CTkButton(
            file_menu,
            text="💾 Save",
            width=70,
            command=self.save_project,
            fg_color="gray",
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            file_menu,
            text="📂 Load",
            width=70,
            command=self.load_project,
            fg_color="gray",
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            file_menu,
            text="📄 Export PDF Report",
            command=self.export_report,
            fg_color="#E67E22",
            hover_color="#D35400",
        ).pack(side="right", padx=5, fill="x", expand=True)

        self.right_frame = ctk.CTkScrollableFrame(self)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        self.canvas_widgets = []
        self.generated_figs = []

    def toggle_theme(self):
        if self.switch_theme.get():
            ctk.set_appearance_mode("dark")
            self.switch_theme.configure(text="🌙 Dark Mode")
        else:
            ctk.set_appearance_mode("light")
            self.switch_theme.configure(text="☀️ Light Mode")

    def sync_and_refresh(self, _event=None):
        self.sync_data_from_ui()
        self.refresh_ui()

    def refresh_ui(self):
        self.draw_project_tab()
        self.draw_geometry_tab()
        self.draw_loads_tab()

    def draw_project_tab(self):
        for widget in self.tab_proj.winfo_children():
            widget.destroy()

        ctk.CTkLabel(
            self.tab_proj,
            text="📋 ข้อมูลโครงการ (Project Info)",
            font=("Helvetica", 14, "bold"),
        ).pack(pady=10)

        frame = ctk.CTkFrame(self.tab_proj, fg_color="transparent")
        frame.pack(fill="x", padx=10, pady=5)

        self.proj_entries = {}
        fields = [
            ("Project Name:", "name"),
            ("Location:", "location"),
            ("Engineer:", "engineer"),
            ("Date:", "date"),
            ("Beam Name/Ref:", "beam_name"),
        ]

        for i, (label_txt, key) in enumerate(fields):
            ctk.CTkLabel(frame, text=label_txt, font=("Helvetica", 12, "bold")).grid(
                row=i, column=0, sticky="e", pady=5, padx=5
            )
            ent = ctk.CTkEntry(frame, width=220)
            ent.insert(0, self.project_data.get(key, ""))
            ent.grid(row=i, column=1, pady=5, padx=5, sticky="w")
            self.proj_entries[key] = ent

        ctk.CTkLabel(self.tab_proj, text="📐 หน่วยที่ใช้ (Units)", font=("Helvetica", 13, "bold")).pack(pady=(15,5))
        unit_frame = ctk.CTkFrame(self.tab_proj, fg_color="transparent")
        unit_frame.pack(fill="x", padx=10)
        ctk.CTkLabel(unit_frame, text="Force:", width=60).grid(row=0, column=0, sticky="e", padx=5, pady=4)
        def _on_unit_change(v, key):
            setattr(self, key, v)
            self._save_settings()
            self.refresh_ui()  # update input labels

        self.unit_force_var = ctk.StringVar(value=self.unit_force)
        ctk.CTkOptionMenu(unit_frame, values=list(UNIT_FORCE_TO_KN.keys()), variable=self.unit_force_var, width=100,
            command=lambda v: _on_unit_change(v, "unit_force")).grid(row=0, column=1, sticky="w", padx=5, pady=4)
        ctk.CTkLabel(unit_frame, text="Length:", width=60).grid(row=1, column=0, sticky="e", padx=5, pady=4)
        self.unit_length_var = ctk.StringVar(value=self.unit_length)
        ctk.CTkOptionMenu(unit_frame, values=list(UNIT_LENGTH_TO_M.keys()), variable=self.unit_length_var, width=100,
            command=lambda v: _on_unit_change(v, "unit_length")).grid(row=1, column=1, sticky="w", padx=5, pady=4)
        ctk.CTkLabel(unit_frame, text="(E modulus always in GPa)", font=("Helvetica", 9), text_color="gray").grid(
            row=2, column=0, columnspan=2, pady=2)

    def draw_geometry_tab(self):
        for widget in self.tab_geo.winfo_children():
            widget.destroy()

        self.tab_geo.grid_columnconfigure(0, weight=1)
        self.tab_geo.grid_rowconfigure(1, weight=1)
        self.tab_geo.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(
            self.tab_geo,
            text="🛠️ ข้อมูลช่วงคาน วัสดุ และหน้าตัด",
            font=("Helvetica", 14, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(5, 5))

        self.span_entries = []
        span_frame = ctk.CTkScrollableFrame(self.tab_geo, height=250)
        span_frame.grid(row=1, column=0, sticky="nsew", padx=5)

        for i, span in enumerate(self.spans_data):
            span_box = ctk.CTkFrame(span_frame, fg_color=("gray85", "gray25"))
            span_box.pack(fill="x", pady=4, padx=2)

            row1 = ctk.CTkFrame(span_box, fg_color="transparent")
            row1.pack(fill="x", pady=2, padx=5)

            ctk.CTkLabel(
                row1,
                text=f"Span {i+1}:",
                font=("Helvetica", 13, "bold"),
                text_color="#3498DB",
            ).pack(side="left", padx=(0, 10))

            ctk.CTkLabel(row1, text=f"L({self.unit_length}):").pack(side="left")
            ent_L = ctk.CTkEntry(row1, width=50)
            ent_L.insert(0, str(span.get("L", 5.0)))
            ent_L.pack(side="left", padx=2)

            row2 = ctk.CTkFrame(span_box, fg_color="transparent")
            row2.pack(fill="x", pady=2, padx=5)

            ctk.CTkLabel(row2, text="Mat:").pack(side="left", padx=(10, 2))
            mat_var = ctk.StringVar(value=span.get("Material", "Concrete"))
            opt_mat = ctk.CTkOptionMenu(
                row2,
                values=["Steel", "Concrete", "Wood", "Aluminum", "Custom E"],
                variable=mat_var,
                width=100,
                command=self.sync_and_refresh,
            )
            opt_mat.pack(side="left", padx=2)

            ent_E = None
            if mat_var.get() == "Custom E":
                ctk.CTkLabel(row2, text="E(GPa):").pack(side="left", padx=(5, 2))
                ent_E = ctk.CTkEntry(row2, width=50)
                ent_E.insert(0, str(span.get("E_custom", 200.0)))
                ent_E.pack(side="left", padx=2)

            row3 = ctk.CTkFrame(span_box, fg_color="transparent")
            row3.pack(fill="x", pady=2, padx=5)

            ctk.CTkLabel(row3, text="Sec:").pack(side="left", padx=(10, 2))
            sec_var = ctk.StringVar(value=span.get("Section", "Rect (bxh)"))
            opt_sec = ctk.CTkOptionMenu(
                row3,
                values=["Rect (bxh)", "Custom I"],
                variable=sec_var,
                width=100,
                command=self.sync_and_refresh,
            )
            opt_sec.pack(side="left", padx=2)

            ent_b = None
            ent_h = None
            ent_I = None

            if sec_var.get() == "Rect (bxh)":
                ctk.CTkLabel(row3, text=f"b({self.unit_length}):").pack(side="left", padx=(5, 2))
                ent_b = ctk.CTkEntry(row3, width=45)
                ent_b.insert(0, str(span.get("b", 0.2)))
                ent_b.pack(side="left", padx=2)

                ctk.CTkLabel(row3, text=f"h({self.unit_length}):").pack(side="left", padx=(5, 2))
                ent_h = ctk.CTkEntry(row3, width=45)
                ent_h.insert(0, str(span.get("h", 0.4)))
                ent_h.pack(side="left", padx=2)
            else:
                ctk.CTkLabel(row3, text=f"I({self.unit_length}⁴):").pack(side="left", padx=(5, 2))
                ent_I = ctk.CTkEntry(row3, width=70)
                ent_I.insert(0, str(span.get("I_custom", 0.001333)))
                ent_I.pack(side="left", padx=2)

            self.span_entries.append(
                {
                    "L": ent_L,
                    "Material": mat_var,
                    "E_custom": ent_E,
                    "Section": sec_var,
                    "b": ent_b,
                    "h": ent_h,
                    "I_custom": ent_I,
                }
            )

        btn_frame = ctk.CTkFrame(self.tab_geo, fg_color="transparent")
        btn_frame.grid(row=2, column=0, sticky="ew", pady=5)
        ctk.CTkButton(
            btn_frame, text="➕ เพิ่มคาน", width=120, command=self.add_span
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            btn_frame,
            text="🗑️ ลบคานล่าสุด",
            width=120,
            fg_color="#C0392B",
            hover_color="#922B21",
            command=self.remove_span,
        ).pack(side="left", padx=5)

        ctk.CTkLabel(
            self.tab_geo,
            text="🏛️ ข้อมูลจุดรองรับ (Supports)",
            font=("Helvetica", 14, "bold"),
        ).grid(row=3, column=0, sticky="w", pady=(15, 5))

        self.support_entries = []
        supp_frame = ctk.CTkScrollableFrame(self.tab_geo, height=180)
        supp_frame.grid(row=4, column=0, sticky="nsew", padx=5, pady=(0, 10))

        options = ["Free", "Pinned", "Roller", "Fixed"]
        for i, supp in enumerate(self.supports_data):
            row = ctk.CTkFrame(supp_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)

            ctk.CTkLabel(row, text=f"Node {i+1}:", font=("Helvetica", 13, "bold")).pack(
                side="left", padx=(5, 10)
            )
            var = ctk.StringVar(value=supp)
            opt = ctk.CTkOptionMenu(row, values=options, variable=var, width=220)
            opt.pack(side="left")
            self.support_entries.append(var)

    def add_span(self):
        self.sync_data_from_ui()
        self.spans_data.append(
            {
                "L": 5.0,
                "Material": "Concrete",
                "E_custom": 200.0,
                "Section": "Rect (bxh)",
                "b": 0.2,
                "h": 0.4,
                "I_custom": 0.001333,
            }
        )
        self.supports_data.append("Roller")
        self.refresh_ui()

    def remove_span(self):
        self.sync_data_from_ui()
        if len(self.spans_data) > 1:
            self.spans_data.pop()
            self.supports_data.pop()
            self.loads_data = [
                l for l in self.loads_data if l["span_index"] < len(self.spans_data)
            ]
            self.refresh_ui()

    def draw_loads_tab(self):
        for widget in self.tab_load.winfo_children():
            widget.destroy()

        self.tab_load.grid_columnconfigure(0, weight=1)
        self.tab_load.grid_rowconfigure(2, weight=1)

        factor_frame = ctk.CTkFrame(self.tab_load)
        factor_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 15))
        ctk.CTkLabel(
            factor_frame, text="⚖️ Load Factors", font=("Helvetica", 14, "bold")
        ).pack(pady=5)

        frow = ctk.CTkFrame(factor_frame, fg_color="transparent")
        frow.pack(pady=5)
        ctk.CTkLabel(
            frow, text="DL:", text_color="#F1C40F", font=("Helvetica", 13, "bold")
        ).pack(side="left")
        self.ent_dl = ctk.CTkEntry(frow, width=40)
        self.ent_dl.insert(0, str(self.load_factors["DL"]))
        self.ent_dl.pack(side="left", padx=5)

        ctk.CTkLabel(
            frow, text="LL:", text_color="#1ABC9C", font=("Helvetica", 13, "bold")
        ).pack(side="left", padx=(15, 0))
        self.ent_ll = ctk.CTkEntry(frow, width=40)
        self.ent_ll.insert(0, str(self.load_factors["LL"]))
        self.ent_ll.pack(side="left", padx=5)

        ctk.CTkLabel(
            self.tab_load,
            text="📦 กำหนดแรง (แยกสี DL/LL เพื่อดูง่าย)",
            font=("Helvetica", 14, "bold"),
        ).grid(row=1, column=0, sticky="w", padx=5)

        self.load_entries = []
        loads_frame = ctk.CTkScrollableFrame(self.tab_load)
        loads_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

        span_options = [f"Span {i+1}" for i in range(len(self.spans_data))]
        type_options = ["Uniform", "Point", "Moment"]
        case_options = ["DL", "LL"]

        # Table Header
        header = ctk.CTkFrame(loads_frame, fg_color="transparent")
        header.pack(fill="x", pady=(0, 2), padx=2)
        ctk.CTkLabel(header, text="Type", width=80).pack(side="left", padx=(5, 2))
        ctk.CTkLabel(header, text="DL/LL", width=55).pack(side="left", padx=2)
        ctk.CTkLabel(header, text="Span", width=75).pack(side="left", padx=2)
        ctk.CTkLabel(header, text="Val", width=50).pack(side="left", padx=2)
        ctk.CTkLabel(header, text="Pos.x", width=40).pack(side="left", padx=2)

        for i, ld in enumerate(self.loads_data):
            bg_color = (
                ("#F9E79F", "#3E3B22") if ld["case"] == "DL" else ("#A3E4D7", "#1E3B33")
            )
            row_box = ctk.CTkFrame(loads_frame, fg_color=bg_color)
            row_box.pack(fill="x", pady=2, padx=2)

            type_var = ctk.StringVar(value=ld["type"])
            ctk.CTkOptionMenu(
                row_box, values=type_options, variable=type_var, width=80
            ).pack(side="left", padx=(5, 2), pady=4)

            case_var = ctk.StringVar(value=ld["case"])
            ctk.CTkOptionMenu(
                row_box,
                values=case_options,
                variable=case_var,
                width=55,
                command=self.sync_and_refresh,
            ).pack(side="left", padx=2, pady=4)

            span_idx = (
                ld["span_index"]
                if ld["span_index"] < len(self.spans_data)
                else len(self.spans_data) - 1
            )
            span_var = ctk.StringVar(value=f"Span {span_idx+1}")
            ctk.CTkOptionMenu(
                row_box, values=span_options, variable=span_var, width=75
            ).pack(side="left", padx=2, pady=4)

            ent_val = ctk.CTkEntry(row_box, width=50)
            ent_val.insert(0, str(ld["val"]))
            ent_val.pack(side="left", padx=2, pady=4)

            ent_x = ctk.CTkEntry(row_box, width=40)
            ent_x.insert(0, str(ld.get("x_offset", 0.0)))
            ent_x.pack(side="left", padx=2, pady=4)

            btn_del = ctk.CTkButton(
                row_box,
                text="❌",
                width=25,
                fg_color="transparent",
                hover_color="#C0392B",
                text_color="red",
                font=("Helvetica", 14),
                command=lambda idx=i: self.delete_load(idx),
            )
            btn_del.pack(side="right", padx=5, pady=4)

            self.load_entries.append(
                {
                    "type": type_var,
                    "case": case_var,
                    "span": span_var,
                    "val": ent_val,
                    "x": ent_x,
                }
            )

        ctk.CTkButton(
            self.tab_load, text="➕ เพิ่มน้ำหนักบรรทุก (Add Load)", command=self.add_load
        ).grid(row=3, column=0, pady=10)

    def add_load(self):
        self.sync_data_from_ui()
        self.loads_data.append(
            {
                "type": "Point",
                "case": "LL",
                "span_index": 0,
                "val": -20.0,
                "x_offset": 2.5,
            }
        )
        self.refresh_ui()

    def delete_load(self, idx):
        self.sync_data_from_ui()
        self.loads_data.pop(idx)
        self.refresh_ui()

    def sync_data_from_ui(self):
        new_spans = []
        for ent in getattr(self, "span_entries", []):
            try:
                L = float(ent["L"].get())
                mat = ent["Material"].get()
                E_cust = float(ent["E_custom"].get()) if ent["E_custom"] else 200.0
                sec = ent["Section"].get()
                b_val = float(ent["b"].get()) if ent["b"] else 0.2
                h_val = float(ent["h"].get()) if ent["h"] else 0.4
                I_cust = float(ent["I_custom"].get()) if ent["I_custom"] else 0.001333
                new_spans.append(
                    {
                        "L": L,
                        "Material": mat,
                        "E_custom": E_cust,
                        "Section": sec,
                        "b": b_val,
                        "h": h_val,
                        "I_custom": I_cust,
                    }
                )
            except ValueError:
                new_spans.append(
                    {
                        "L": 5.0,
                        "Material": "Concrete",
                        "E_custom": 200.0,
                        "Section": "Rect (bxh)",
                        "b": 0.2,
                        "h": 0.4,
                        "I_custom": 0.001333,
                    }
                )
        if new_spans:
            self.spans_data = new_spans

        if getattr(self, "support_entries", []):
            self.supports_data = [var.get() for var in self.support_entries]

        if hasattr(self, "ent_dl"):
            try:
                self.load_factors["DL"] = float(self.ent_dl.get())
            except ValueError:
                pass
            try:
                self.load_factors["LL"] = float(self.ent_ll.get())
            except ValueError:
                pass

        new_loads = []
        for ld in getattr(self, "load_entries", []):
            try:
                val = float(ld["val"].get())
                x_off = float(ld["x"].get())
                span_idx = int(ld["span"].get().replace("Span ", "")) - 1
                new_loads.append(
                    {
                        "type": ld["type"].get(),
                        "case": ld["case"].get(),
                        "span_index": span_idx,
                        "val": val,
                        "x_offset": x_off,
                    }
                )
            except ValueError:
                pass
        if getattr(self, "load_entries", []):
            self.loads_data = new_loads

    def save_project(self):
        self.sync_data_from_ui()
        data = {
            "spans": self.spans_data,
            "supports": self.supports_data,
            "loads": self.loads_data,
            "factors": self.load_factors,
            "unit_force": self.unit_force,
            "unit_length": self.unit_length,
        }
        fpath = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON Files", "*.json")]
        )
        if fpath:
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            messagebox.showinfo("Success", "Project Saved!")

    def load_project(self):
        fpath = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if fpath:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.spans_data = data.get("spans", [{"L": 5.0, "Material": "Concrete"}])
            self.supports_data = data.get("supports", ["Pinned", "Roller"])
            self.loads_data = data.get("loads", [])
            self.load_factors = data.get("factors", {"DL": 1.4, "LL": 1.7})
            self.unit_force = data.get("unit_force", "kN")
            self.unit_length = data.get("unit_length", "m")
            self.refresh_ui()

    def export_report(self):
        if self.ss is None:
            messagebox.showwarning(
                "ยังไม่มีข้อมูล", "กรุณากดปุ่ม 🚀 อัปเดตโมเดล & วิเคราะห์ ก่อนสั่งพิมพ์รายงาน!"
            )
            return

        fpath = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=[("PDF Documents", "*.pdf")]
        )
        if not fpath:
            return

        try:
            import datetime
            import io

            import numpy as np
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import (ParagraphStyle,
                                              getSampleStyleSheet)
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.platypus import Image as RLImage
            from reportlab.platypus import (Paragraph, SimpleDocTemplate,
                                            Spacer, Table, TableStyle)

            # Try to register Tahoma for Thai support, fallback to Helvetica
            font_name = "Helvetica"
            try:
                import os

                tahoma_path = r"C:\Windows\Fonts\tahoma.ttf"
                if os.path.exists(tahoma_path):
                    pdfmetrics.registerFont(TTFont("Tahoma", tahoma_path))
                    # Register bold for headers
                    tahoma_b_path = r"C:\Windows\Fonts\tahomabd.ttf"
                    if os.path.exists(tahoma_b_path):
                        pdfmetrics.registerFont(TTFont("Tahoma-Bold", tahoma_b_path))
                    font_name = "Tahoma"
            except:
                pass

            b_font = font_name + "-Bold" if font_name == "Tahoma" else "Helvetica-Bold"

            doc = SimpleDocTemplate(
                fpath,
                pagesize=A4,
                rightMargin=30,
                leftMargin=30,
                topMargin=30,
                bottomMargin=30,
            )
            elements = []
            styles = getSampleStyleSheet()

            # Custom Styles
            title_style = ParagraphStyle(
                "TitleStyle",
                parent=styles["Heading1"],
                fontName=b_font,
                fontSize=18,
                textColor=colors.HexColor("#1ABC9C"),
                alignment=1,
                spaceAfter=6,
            )
            subtitle_style = ParagraphStyle(
                "SubTitle",
                parent=styles["Normal"],
                fontName=font_name,
                fontSize=10,
                textColor=colors.gray,
                alignment=1,
                spaceAfter=20,
            )
            h2_style = ParagraphStyle(
                "H2",
                parent=styles["Heading2"],
                fontName=b_font,
                fontSize=12,
                textColor=colors.HexColor("#2C3E50"),
                spaceBefore=15,
                spaceAfter=5,
            )
            normal_style = ParagraphStyle(
                "NormalStyle", parent=styles["Normal"], fontName=font_name, fontSize=9
            )

            # 1. Header
            elements.append(Paragraph("GO ContinuousBeamAnalyzerPro", title_style))
            p_date = self.project_data.get(
                "date", datetime.datetime.now().strftime("%Y-%m-%d")
            )
            elements.append(
                Paragraph(
                    f"Designed by: {self.project_data.get('engineer', 'N/A')} | Date: {p_date}",
                    subtitle_style,
                )
            )

            # 2. Project Info Table
            elements.append(Paragraph("Project Information", h2_style))
            proj_data = [
                [
                    "Project Name:",
                    self.project_data.get("name", "-"),
                    "Location:",
                    self.project_data.get("location", "-"),
                ],
                ["Beam Name:", self.project_data.get("beam_name", "-"), "", ""],
            ]
            t_proj = Table(proj_data, colWidths=[80, 180, 80, 180])
            t_proj.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, -1), font_name),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("TEXTCOLOR", (0, 0), (0, -1), colors.darkblue),
                        ("TEXTCOLOR", (2, 0), (2, -1), colors.darkblue),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            elements.append(t_proj)
            elements.append(Spacer(1, 10))

            # Helper function for matplotlib to Platypus Image
            def fig_to_image(fig, width_ratio=0.9):
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                img = RLImage(buf)
                # Scale to fit A4 (usable width ~535 pt, usable height ~780 pt)
                usable_width = 535 * width_ratio
                max_height = 220  # cap height so images fit on page
                aspect = img.drawHeight / float(img.drawWidth)
                img.drawWidth = usable_width
                img.drawHeight = usable_width * aspect
                if img.drawHeight > max_height:
                    img.drawHeight = max_height
                    img.drawWidth = max_height / aspect
                plt.close(fig)
                return img

            def boost_fonts(fig, base=12):
                for ax in fig.get_axes():
                    ax.title.set_fontsize(base + 2)
                    ax.xaxis.label.set_fontsize(base)
                    ax.yaxis.label.set_fontsize(base)
                    ax.tick_params(axis="both", labelsize=base)
                    for text in ax.texts:
                        text.set_fontsize(base)
                fig.suptitle(fig.texts[0].get_text() if fig.texts else "", fontsize=base + 3, fontweight="bold") if fig.texts else None

            ff = UNIT_FORCE_TO_KN[self.unit_force]
            lf = UNIT_LENGTH_TO_M[self.unit_length]

            # 3. Structure Image
            elements.append(
                Paragraph("Structure Model (Applied & Factored Loads)", h2_style)
            )
            struct_fig = self.ss.show_structure(show=False)
            self.scale_support_symbols(struct_fig)
            self._rescale_graph_labels(struct_fig, "structure")
            self.add_dimensions(struct_fig)
            struct_fig.set_size_inches(10, 3.5)
            struct_fig.tight_layout(pad=0.8)
            boost_fonts(struct_fig)
            elements.append(fig_to_image(struct_fig, 1.0))
            elements.append(Spacer(1, 10))

            # 4. Geometry Table
            elements.append(Paragraph("Geometry & Sections", h2_style))
            geo_data = [["Span", f"Length ({self.unit_length})", "Material", "Section", "Left Support"]]
            for i, span in enumerate(self.spans_data):
                supp = self.supports_data[i]
                mat = span.get("Material", "Concrete")
                E = span.get("E_custom", 200) if mat == "Custom E" else mat
                sec = span.get("Section", "Rect")
                if sec == "Rect (bxh)":
                    b = span.get("b", 0.20)
                    h = span.get("h", 0.40)
                    sec_str = f"Rect ({b:.2f}x{h:.2f})"
                else:
                    sec_str = sec
                geo_data.append([str(i + 1), f"{span['L']:.2f}", str(E), sec_str, supp])
            geo_data.append(
                ["-", "-", "-", "-", f"Right Node: {self.supports_data[-1]}"]
            )

            t_geo = Table(geo_data, colWidths=[40, 80, 120, 120, 100])
            t_geo.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D5D8DC")),
                        ("FONTNAME", (0, 0), (-1, 0), b_font),
                        ("FONTNAME", (0, 1), (-1, -1), font_name),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.gray),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            elements.append(t_geo)
            elements.append(Spacer(1, 10))

            # 5. Loads Table
            combo_str = f"{self.load_factors.get('DL', 1.4)}DL + {self.load_factors.get('LL', 1.7)}LL"
            elements.append(
                Paragraph(f"Applied Loads (Combination = {combo_str})", h2_style)
            )
            ld_data = [
                ["Type", "DL/LL", "Span", f"Pos x ({self.unit_length})", f"Value ({self.unit_force})", f"Factored ({self.unit_force})"]
            ]
            for ld in self.loads_data:
                fac = self.load_factors[ld["case"]]
                val_factored = ld["val"] * fac
                ld_data.append(
                    [
                        ld["type"],
                        ld["case"],
                        str(ld["span_index"] + 1),
                        f"{ld.get('x_offset',0):.2f}",
                        f"{ld['val']:.3f}",
                        f"{val_factored:.3f}",
                    ]
                )
            if len(ld_data) == 1:
                ld_data.append(["-", "-", "-", "-", "-", "-"])

            t_ld = Table(ld_data, colWidths=[80, 60, 60, 80, 80, 80])
            t_ld.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D5D8DC")),
                        ("FONTNAME", (0, 0), (-1, 0), b_font),
                        ("FONTNAME", (0, 1), (-1, -1), font_name),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.gray),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            elements.append(t_ld)
            elements.append(Spacer(1, 15))

            # 6. Analysis Results Table
            elements.append(Paragraph("Beam Span Max/Min Results", h2_style))

            fig_defl = self.ss.show_displacement(factor=1, show=False)
            ax_defl = fig_defl.gca()
            element_max_defl = []
            if ax_defl.lines:
                for line in ax_defl.lines:
                    yd = np.array(line.get_ydata(), dtype=float)
                    if len(yd) >= 8:
                        element_max_defl.append(float(np.max(np.abs(yd))) / lf)
            plt.close(fig_defl)

            elem_res = self.ss.get_element_results()
            res_data = [["Span", f"+M ({self.unit_force}-{self.unit_length})", f"-M ({self.unit_force}-{self.unit_length})", f"V max ({self.unit_force})", f"Defl ({self.unit_length})"]]
            for span_idx in range(len(self.spans_data)):
                mapped_elems = self.span_elem_map.get(span_idx, [])
                if not mapped_elems:
                    continue
                max_M, min_M, max_V, max_D = -99999, 99999, 0, 0.0
                for el_id in mapped_elems:
                    el = elem_res[el_id - 1]
                    max_M = max(max_M, max(el["Mmin"] / (ff * lf), el["Mmax"] / (ff * lf)))
                    min_M = min(min_M, min(el["Mmin"] / (ff * lf), el["Mmax"] / (ff * lf)))
                    max_V = max(max_V, max(abs(el["Qmin"]) / ff, abs(el["Qmax"]) / ff))
                    if (el_id - 1) < len(element_max_defl):
                        max_D = max(max_D, element_max_defl[el_id - 1])
                if max_M == -99999:
                    max_M = 0
                if min_M == 99999:
                    min_M = 0

                res_data.append(
                    [
                        str(span_idx + 1),
                        f"{max_M:.3f}",
                        f"{min_M:.3f}",
                        f"{max_V:.3f}",
                        f"{max_D:.5f}",
                    ]
                )

            t_res = Table(res_data, colWidths=[60, 100, 100, 100, 100])
            t_res.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D5D8DC")),
                        ("FONTNAME", (0, 0), (-1, 0), b_font),
                        ("FONTNAME", (0, 1), (-1, -1), font_name),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.gray),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            elements.append(t_res)
            elements.append(Spacer(1, 15))

            abs_max_deflection = max(element_max_defl) if element_max_defl else 0.0
            elements.append(
                Paragraph(
                    f"<b>Overall Max Deflection:</b> {abs_max_deflection:.5f} {self.unit_length}",
                    normal_style,
                )
            )
            elements.append(Spacer(1, 15))

            # 7. Reaction Forces Table
            elements.append(Paragraph("Reaction Forces", h2_style))
            reacts = self.ss.get_node_results_system()
            span_boundaries_rep = [0.0]
            for span in self.spans_data:
                lf_rep = UNIT_LENGTH_TO_M[self.unit_length]
                span_boundaries_rep.append(span_boundaries_rep[-1] + span["L"] * lf_rep)
            react_table_data = [["Node", f"x ({self.unit_length})", "Support", f"Fy ({self.unit_force})", f"M ({self.unit_force}·{self.unit_length})"]]
            for i, x_bound in enumerate(span_boundaries_rep):
                for r in reacts:
                    node = self.ss.node_map.get(r["id"])
                    if node and abs(node.vertex.x - x_bound) < 0.001:
                        Fy_disp = round(r.get("Fy", 0.0) / ff, 3)
                        M_disp = round(r.get("Ty", 0.0) / (ff * lf), 3)
                        supp_type = self.supports_data[i] if i < len(self.supports_data) else "-"
                        react_table_data.append([
                            str(i + 1),
                            f"{span["L"] if False else x_bound / lf_rep:.2f}",
                            supp_type,
                            f"{Fy_disp:.3f}",
                            f"{M_disp:.3f}" if abs(M_disp) > 0.001 else "0.000"
                        ])
                        break
            t_react = Table(react_table_data, colWidths=[45, 70, 80, 100, 100])
            t_react.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D5D8DC")),
                ("FONTNAME", (0, 0), (-1, 0), b_font),
                ("FONTNAME", (0, 1), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.gray),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]))
            elements.append(t_react)
            elements.append(Spacer(1, 10))

            # 7b. Reaction Forces Diagram
            react_fig = self.ss.show_reaction_force(show=False)
            self.scale_support_symbols(react_fig)
            self._rescale_graph_labels(react_fig, "reaction")
            self.add_dimensions(react_fig)
            react_fig.set_size_inches(10, 3.0)
            react_fig.tight_layout(pad=0.8)
            boost_fonts(react_fig)
            elements.append(fig_to_image(react_fig, 0.9))
            elements.append(Spacer(1, 10))

            # 8. SFD
            elements.append(Paragraph("Shear Force Diagram (SFD)", h2_style))
            sfd_fig = self.ss.show_shear_force(show=False)
            self.scale_support_symbols(sfd_fig)
            self._rescale_graph_labels(sfd_fig, "sfd")
            self.add_dimensions(sfd_fig)
            self.colorize_diagram(sfd_fig, is_bmd=False)
            boost_fonts(sfd_fig)
            elements.append(fig_to_image(sfd_fig, 0.9))
            elements.append(Spacer(1, 10))

            # 9. BMD
            elements.append(Paragraph("Bending Moment Diagram (BMD)", h2_style))
            bmd_fig = self.ss.show_bending_moment(show=False)
            self.scale_support_symbols(bmd_fig)
            self._rescale_graph_labels(bmd_fig, "bmd")
            self.add_dimensions(bmd_fig)
            self.colorize_diagram(bmd_fig, is_bmd=True)
            boost_fonts(bmd_fig)
            elements.append(fig_to_image(bmd_fig, 0.9))
            elements.append(Spacer(1, 10))

            # 10. Deflection Diagram
            elements.append(Paragraph("Deflection Diagram", h2_style))
            defl_fig = self.ss.show_displacement(show=False)
            fig_true = self.ss.show_displacement(factor=1, show=False)
            ax_true = fig_true.gca()
            ax_scaled = defl_fig.gca()
            if ax_true.lines and ax_scaled.lines:
                for text in ax_scaled.texts:
                    tx, ty = text.get_position()
                    best_dist = float("inf")
                    true_val = 0.0
                    for line in ax_true.lines:
                        xd = line.get_xdata()
                        yd = line.get_ydata()
                        if len(xd) < 8:  # skip support symbol lines
                            continue
                        for i, xv in enumerate(xd):
                            if abs(xv - tx) < best_dist:
                                best_dist = abs(xv - tx)
                                true_val = yd[i]
                    text.set_text(f"{true_val / lf:.5f} {self.unit_length}")
            plt.close(fig_true)
            self.add_dimensions(defl_fig)
            boost_fonts(defl_fig)
            self.scale_support_symbols(defl_fig)
            elements.append(fig_to_image(defl_fig, 0.9))

            def draw_footer(canvas, doc_obj):
                canvas.saveState()
                canvas.setFont(font_name, 9)
                canvas.setFillColor(colors.gray)
                proj_name = self.project_data.get("name", "Continuous Beam Analysis")
                footer_text = f"Project: {proj_name}   |   Page {doc_obj.page}"
                canvas.drawRightString(A4[0] - 30, 20, footer_text)
                canvas.restoreState()

            # Build PDF
            doc.build(elements, onFirstPage=draw_footer, onLaterPages=draw_footer)

            messagebox.showinfo(
                "Success", f"ส่งออกรายงานสำเร็จ!\nถูกบันทึกเป็น PDF ที่:\n{fpath}"
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            messagebox.showerror("Export Error", f"ไม่สามารถสร้างไฟล์ PDF ได้: {e}")

    # ==========================================================
    # SOLVER CORE (anaStruct)
    # ==========================================================
    def _calculate_EI(self, span_prop, lf=1.0):
        mat = span_prop.get("Material", "Concrete")
        if mat == "Steel":
            E = 200e6
        elif mat == "Concrete":
            E = 25e6
        elif mat == "Wood":
            E = 10e6
        elif mat == "Aluminum":
            E = 70e6
        else:
            try:
                E = float(span_prop.get("E_custom", 200)) * 1e6
            except:
                E = 25e6

        sec = span_prop.get("Section", "Rect (bxh)")
        if sec == "Rect (bxh)":
            b = float(span_prop.get("b", 0.2)) * lf
            h = float(span_prop.get("h", 0.4)) * lf
            I = (b * h**3) / 12.0
            A = b * h
        else:
            I = float(span_prop.get("I_custom", 0.001333)) * lf**4
            A = 0.1 * lf**2
        return E, I, A

    def calculate(self):
        self.sync_data_from_ui()
        ff = UNIT_FORCE_TO_KN[self.unit_force]
        lf = UNIT_LENGTH_TO_M[self.unit_length]

        span_boundaries = [0.0]
        for span in self.spans_data:
            span_boundaries.append(span_boundaries[-1] + span["L"] * lf)

        all_nodes_x = set(span_boundaries)
        for ld in self.loads_data:
            if ld["type"] in ["Point", "Moment"]:
                x_abs = span_boundaries[ld["span_index"]] + ld["x_offset"] * lf
                x_abs = max(
                    span_boundaries[ld["span_index"]],
                    min(span_boundaries[ld["span_index"] + 1], x_abs),
                )
                all_nodes_x.add(x_abs)
                ld["x_abs"] = x_abs

        sorted_x = sorted(list(all_nodes_x))

        self.ss = SystemElements()
        self.span_elem_map = {i: [] for i in range(len(self.spans_data))}
        elem_id = 1

        for i in range(len(sorted_x) - 1):
            L_elem = sorted_x[i + 1] - sorted_x[i]
            x_mid = sorted_x[i] + L_elem / 2.0

            orig_span_idx = 0
            for j in range(len(span_boundaries) - 1):
                if (
                    span_boundaries[j] - 0.001
                    <= x_mid
                    <= span_boundaries[j + 1] + 0.001
                ):
                    orig_span_idx = j
                    break

            self.span_elem_map[orig_span_idx].append(elem_id)
            elem_id += 1

            span_prop = self.spans_data[orig_span_idx]
            E, I, A = self._calculate_EI(span_prop, lf)
            self.ss.add_element(
                location=[[sorted_x[i], 0], [sorted_x[i + 1], 0]], EA=E * A, EI=E * I
            )

        for i, x_bound in enumerate(span_boundaries):
            node_id = sorted_x.index(x_bound) + 1
            supp = self.supports_data[i]
            if supp == "Pinned":
                self.ss.add_support_hinged(node_id=node_id)
            elif supp == "Roller":
                self.ss.add_support_roll(node_id=node_id, direction=2)
            elif supp == "Fixed":
                self.ss.add_support_fixed(node_id=node_id)

        dl_f = self.load_factors["DL"]
        ll_f = self.load_factors["LL"]

        for ld in self.loads_data:
            factor = dl_f if ld["case"] == "DL" else ll_f
            raw_val = ld["val"] * factor

            if ld["type"] == "Uniform":
                val = raw_val * ff / lf
                x_start = span_boundaries[ld["span_index"]]
                x_end = span_boundaries[ld["span_index"] + 1]
                for i in range(len(sorted_x) - 1):
                    mid_x = (sorted_x[i] + sorted_x[i + 1]) / 2.0
                    if x_start - 0.001 <= mid_x <= x_end + 0.001:
                        self.ss.q_load(element_id=i + 1, q=val)

            elif ld["type"] == "Point":
                val = raw_val * ff
                node_id = sorted_x.index(ld["x_abs"]) + 1
                self.ss.point_load(node_id=node_id, Fy=val)

            elif ld["type"] == "Moment":
                val = raw_val * ff * lf
                node_id = sorted_x.index(ld["x_abs"]) + 1
                self.ss.moment_load(node_id=node_id, Ty=val)

        try:
            self.ss.solve()
            self.show_results_table(span_boundaries)
            self.update_plots()
        except Exception as e:
            messagebox.showerror(
                "Solver Error",
                f"เกิดข้อผิดพลาดในการวิเคราะห์\nอาจเป็นเพราะการกำหนดจุดรองรับไม่สมบูรณ์ หรือโครงสร้างแกว่งอิสระ\n\nรายละเอียด: {e}",
            )

    # ==========================================================
    # RESULTS TABLE & PLOTS CORE
    # ==========================================================
    def show_results_table(self, span_boundaries):
        ff = UNIT_FORCE_TO_KN[self.unit_force]
        lf = UNIT_LENGTH_TO_M[self.unit_length]
        fu = self.unit_force
        lu = self.unit_length

        for widget in self.tab_res.winfo_children():
            widget.destroy()

        ctk.CTkLabel(
            self.tab_res,
            text="📊 สรุปผลการวิเคราะห์ (Analysis Results)",
            font=("Helvetica", 16, "bold"),
        ).pack(pady=10)

        self.txt_results = ctk.CTkTextbox(
            self.tab_res, width=400, height=500, font=("Consolas", 13)
        )
        self.txt_results.pack(fill="both", expand=True, padx=10, pady=10)

        result_str = "=================================================\n"
        result_str += " REACTION FORCES (แรงปฏิกิริยาที่จุดรองรับ)\n"
        result_str += "=================================================\n"
        reacts = self.ss.get_node_results_system()

        for i, x_bound in enumerate(span_boundaries):
            for r in reacts:
                if abs(r["id"] - (reacts.index(r) + 1)) == 0:
                    node = self.ss.node_map[r["id"]]
                    if abs(node.vertex.x - x_bound) < 0.001:
                        supp_type = self.supports_data[i]
                        Fy = round(r["Fy"] / ff, 3) if "Fy" in r else 0.0
                        M = round(r["Ty"] / (ff * lf), 3) if "Ty" in r else 0.0
                        result_str += f"[Node {i+1}] x={x_bound/lf:.2f}{lu} ({supp_type}) \n"
                        result_str += f"   Fy = {Fy:<10.3f} {fu}\n"
                        if abs(M) > 0.001:
                            result_str += f"    M = {M:<10.3f} {fu}-{lu}\n"
                        result_str += (
                            "------------------------------------------------\n"
                        )

        result_str += "\n=================================================\n"
        result_str += " BEAM SPAN MAX/MIN SUMMARY (สรุปแยกตามช่วงคาน)\n"
        result_str += "=================================================\n"

        # Calculate max deflection for ALL elements by inspecting the exact curve
        import matplotlib.pyplot as plt
        import numpy as np

        fig_defl = self.ss.show_displacement(factor=1, show=False)
        ax_defl = fig_defl.gca()
        element_max_defl = []
        if ax_defl.lines:
            for line in ax_defl.lines:
                yd = np.array(line.get_ydata(), dtype=float)
                # Filter out support drawing artifacts (triangles/circles) which have few points (usually 2-6)
                if len(yd) >= 8:
                    element_max_defl.append(float(np.max(np.abs(yd))) / lf)
        plt.close(fig_defl)

        elem_res = self.ss.get_element_results()

        result_str += f"{'Span':<6} | {'+M ('+fu+'-'+lu+')':<14} | {'-M ('+fu+'-'+lu+')':<14} | {'V max ('+fu+')':<14} | {'Defl ('+lu+')':<12}\n"
        result_str += "-" * 70 + "\n"

        for span_idx in range(len(self.spans_data)):
            mapped_elems = self.span_elem_map.get(span_idx, [])
            if not mapped_elems:
                continue

            max_M, min_M, max_V, max_D = -99999, 99999, 0, 0.0
            for el_id in mapped_elems:
                el = elem_res[el_id - 1]
                max_M = max(max_M, max(el["Mmin"] / (ff * lf), el["Mmax"] / (ff * lf)))
                min_M = min(min_M, min(el["Mmin"] / (ff * lf), el["Mmax"] / (ff * lf)))
                max_V = max(max_V, max(abs(el["Qmin"]) / ff, abs(el["Qmax"]) / ff))

                # Retrieve deflection mapped to element ID (0-indexed list)
                if (el_id - 1) < len(element_max_defl):
                    max_D = max(max_D, element_max_defl[el_id - 1])

            if max_M == -99999:
                max_M = 0
            if min_M == 99999:
                min_M = 0

            result_str += f"{span_idx+1:<6} | {max_M:<14.3f} | {min_M:<14.3f} | {max_V:<14.3f} | {max_D:<12.5f}\n"

        result_str += "\n=================================================\n"
        result_str += " OVERALL DEFLECTION SUMMARY (ระยะโก่งตัวสูงสุดรวม)\n"
        result_str += "=================================================\n"

        abs_max_deflection = max(element_max_defl) if element_max_defl else 0.0
        result_str += f"Max Deflection: {abs_max_deflection:.5f} {lu}\n"

        self.txt_results.insert("0.0", result_str)
        self.txt_results.configure(state="disabled")

    def colorize_diagram(self, fig, is_bmd=False):
        import numpy as np
        from matplotlib.collections import LineCollection

        ax = fig.gca()

        # 1. Remove existing collection fills
        for c in list(ax.collections):
            c.remove()

        # 2. Find global Y magnitude to set threshold (filter out near-zero baseline/hatch lines)
        all_y_max = max(
            (
                float(np.max(np.abs(line.get_ydata())))
                for line in ax.lines
                if len(line.get_ydata()) >= 2
            ),
            default=0.0,
        )
        if all_y_max < 1e-6:
            return

        y_threshold = (
            all_y_max * 0.05
        )  # Must be >= 5% of max to count as a diagram curve
        x_threshold = (
            sum(s["L"] for s in self.spans_data) if self.spans_data else 1.0
        ) * 0.01

        # 3. Collect REAL diagram curves only.
        # Key insight: anastruct support symbols (rollers, pins, hatch lines) have only 2-6 points.
        # Actual SFD/BMD curves per element have many points (typically 10+).
        diagram_segments = []
        for line in ax.lines:
            xd = np.array(line.get_xdata(), dtype=float)
            yd = np.array(line.get_ydata(), dtype=float)
            if len(xd) < 8:  # Support symbols have < 8 points — skip
                continue
            if np.max(np.abs(yd)) < y_threshold:  # Skip near-zero baselines
                continue
            if (np.max(xd) - np.min(xd)) < x_threshold:  # Skip near-vertical lines
                continue
            diagram_segments.append((xd, yd, line))

        # 4. Fill colors & recolor line with zero-crossing split
        color_pos = "#E74C3C"  # RED  (+V, +M above zero)
        color_neg = "#2980B9"  # BLUE (-V, -M below zero)

        for xd, yd, line in diagram_segments:
            x = xd.copy()
            y = yd.copy()
            for i in range(1, len(x)):
                if x[i] <= x[i - 1]:
                    x[i] = x[i - 1] + 1e-9
            ax.fill_between(
                x,
                0,
                y,
                where=(y >= 0),
                facecolor=color_pos,
                alpha=0.45,
                interpolate=True,
            )
            ax.fill_between(
                x,
                0,
                y,
                where=(y <= 0),
                facecolor=color_neg,
                alpha=0.45,
                interpolate=True,
            )

            # Recolor the line with red above zero / blue below zero
            line.set_visible(False)
            pts = np.array([x, y]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            seg_colors = [
                color_pos if (y[i] + y[i + 1]) / 2 >= 0 else color_neg
                for i in range(len(segs))
            ]
            lc = LineCollection(segs, colors=seg_colors, linewidths=1.5)
            ax.add_collection(lc)

        # 5. Neutral spine colors
        for spine in ax.spines.values():
            spine.set_edgecolor("#BDC3C7")
            spine.set_linewidth(0.8)

    def add_dimensions(self, fig):
        if not self.spans_data:
            return
        import numpy as np
        ax = fig.gca()

        # Compute tight y bounds from all artists (lines, patches, collections)
        y_vals = []
        for line in ax.get_lines():
            yd = np.asarray(line.get_ydata(), dtype=float)
            finite = yd[np.isfinite(yd)]
            if len(finite) > 0:
                y_vals.extend(finite.tolist())
        for patch in ax.patches:
            try:
                verts = patch.get_path().vertices[:, 1]
                finite = verts[np.isfinite(verts)]
                if len(finite) > 0:
                    y_vals.extend(finite.tolist())
            except Exception:
                pass
        for coll in ax.collections:
            try:
                for path in coll.get_paths():
                    verts = path.vertices[:, 1]
                    finite = verts[np.isfinite(verts)]
                    if len(finite) > 0:
                        y_vals.extend(finite.tolist())
            except Exception:
                pass
        # Include text annotation y-positions so labels never clip outside axes
        for text in ax.texts:
            try:
                _, ty = text.get_position()
                if np.isfinite(ty):
                    y_vals.append(float(ty))
            except Exception:
                pass

        if y_vals:
            ymin = min(y_vals)
            ymax = max(y_vals)
        else:
            ymin, ymax = ax.get_ylim()

        # Ensure minimum visible range
        if abs(ymax - ymin) < 0.5:
            ymin -= 0.25
            ymax += 0.25

        y_range = ymax - ymin

        # Dimension line below data
        y_line = ymin - y_range * 0.10

        bot_limit = y_line - y_range * 0.05
        top_limit = ymax + y_range * 0.05
        if top_limit < 0:
            top_limit = y_range * 0.08

        current_x = 0
        for span in self.spans_data:
            L = span["L"]
            mid_x = current_x + (L / 2)
            ax.annotate(
                "",
                xy=(current_x, y_line),
                xytext=(current_x + L, y_line),
                arrowprops=dict(arrowstyle="<|-|>", color="gray", lw=1.5),
            )
            ax.text(
                mid_x,
                y_line + y_range * 0.03,
                f"{L} m",
                ha="center",
                va="bottom",
                fontsize=10,
                color="darkblue",
                fontweight="bold",
                fontfamily="Tahoma",
            )
            ax.plot(
                [current_x, current_x],
                [y_line - y_range * 0.03, y_line + y_range * 0.03],
                color="gray",
                lw=1,
            )
            current_x += L
        ax.plot(
            [current_x, current_x],
            [y_line - y_range * 0.03, y_line + y_range * 0.03],
            color="gray",
            lw=1,
        )

        ax.set_ylim(bot_limit, top_limit)
        x_margin = current_x * 0.04
        ax.set_xlim(-x_margin, current_x + x_margin)

    def scale_support_symbols(self, fig, scale=0.30):
        """Scale down support symbol patches (triangles only — skip fill_between areas)."""
        import numpy as np
        ax = fig.gca()
        for patch in ax.patches:
            try:
                path = patch.get_path()
                verts = path.vertices
                # Support triangles have ≤ 8 vertices; fill_between patches have many more
                if len(verts) < 3 or len(verts) > 8:
                    continue
                centroid = verts.mean(axis=0)
                path.vertices[:] = centroid + (verts - centroid) * scale
            except Exception:
                pass

    def update_plots(self):
        if self.ss is None:
            return

        for widget in self.canvas_widgets:
            widget.destroy()
        self.canvas_widgets.clear()
        self.generated_figs.clear()
        plt.close("all")

        plots_to_show = []
        if self.chk_structure.get():
            fig = self.ss.show_structure(show=False)
            fig.suptitle(
                f"Structure Model (Factored Loads: DL*{self.load_factors['DL']} + LL*{self.load_factors['LL']})",
                fontsize=12,
                fontweight="bold",
                fontfamily="Tahoma",
            )
            self._rescale_graph_labels(fig, "structure")
            plots_to_show.append(fig)  # structure: no symbol scaling

        if self.chk_reaction.get():
            fig = self.ss.show_reaction_force(show=False)
            fig.suptitle(
                "Reaction Forces", fontsize=12, fontweight="bold", fontfamily="Tahoma"
            )
            self.scale_support_symbols(fig)
            self._rescale_graph_labels(fig, "reaction")
            plots_to_show.append(fig)

        if self.chk_sfd.get():
            fig = self.ss.show_shear_force(show=False)
            fig.suptitle(
                "Shear Force Diagram (SFD)",
                fontsize=12,
                fontweight="bold",
                fontfamily="Tahoma",
            )
            self.scale_support_symbols(fig)   # scale BEFORE colorize
            self._rescale_graph_labels(fig, "sfd")
            self.colorize_diagram(fig, is_bmd=False)
            plots_to_show.append(fig)

        if self.chk_bmd.get():
            fig = self.ss.show_bending_moment(show=False)
            fig.suptitle(
                "Bending Moment Diagram (BMD)",
                fontsize=12,
                fontweight="bold",
                fontfamily="Tahoma",
            )
            self.scale_support_symbols(fig)   # scale BEFORE colorize
            self._rescale_graph_labels(fig, "bmd")
            self.colorize_diagram(fig, is_bmd=True)
            plots_to_show.append(fig)

        if self.chk_deflection.get():
            # Create the actual scaled deflection figure
            fig = self.ss.show_displacement(show=False)
            fig.suptitle(
                "Deflection Diagram",
                fontsize=12,
                fontweight="bold",
                fontfamily="Tahoma",
            )

            # AnaStruct rounds deflection texts. We'll replace them by calculating the true deflection.
            import numpy as np

            # 1. Get the true unscaled deflections by plotting with factor=1
            fig_true = self.ss.show_displacement(factor=1, show=False)
            ax_true = fig_true.gca()
            ax_scaled = fig.gca()

            # 2. Update all text annotations by matching X with true deflection curve
            lf = UNIT_LENGTH_TO_M[self.unit_length]
            if ax_true.lines and ax_scaled.lines:
                for text in ax_scaled.texts:
                    # Find the curve point closest to this text's X coordinate
                    tx, ty = text.get_position()

                    # Find the true unscaled deflection at this X
                    best_dist = float("inf")
                    true_val = 0.0

                    for line in ax_true.lines:
                        xd = line.get_xdata()
                        yd = line.get_ydata()
                        if len(xd) < 8:  # skip support symbol lines
                            continue
                        for i, xv in enumerate(xd):
                            if abs(xv - tx) < best_dist:
                                best_dist = abs(xv - tx)
                                true_val = yd[i]

                    # Convert m → user length unit
                    text.set_text(f"{true_val / lf:.5f} {self.unit_length}")

            plt.close(fig_true)
            self.scale_support_symbols(fig)
            plots_to_show.append(fig)

        for fig in plots_to_show:
            self.add_dimensions(fig)

            # Apply x-axis margin: 5% padding each side of total beam length
            total_L = sum(s["L"] for s in self.spans_data)
            x_margin = total_L * 0.05
            ax = fig.gca()
            ax.set_xlim(-x_margin, total_L + x_margin)

            fig.set_size_inches(9, 3.5)
            fig.tight_layout(pad=1.5)
            self.generated_figs.append(fig)

            canvas = FigureCanvasTkAgg(fig, master=self.right_frame)
            canvas.draw()
            toolbar = NavigationToolbar2Tk(canvas, self.right_frame, pack_toolbar=False)
            toolbar.update()

            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill="x", padx=10, pady=(15, 0))
            toolbar.pack(fill="x", padx=10, pady=(0, 15))

            self.canvas_widgets.extend([canvas_widget, toolbar])


if __name__ == "__main__":
    app = ContinuousBeamAnalyzerPro()
    app.mainloop()
