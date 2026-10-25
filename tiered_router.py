"""Tiered Routing Harness: Small Brain (Reflex) / Big Brain (Reflection)

Implements:
1. Tier 1: Low-latency reflex classifier (<20ms SLA) with strict physical schema binding
   and safety abstention rules.
2. Tier 2: General reasoning reflection engine (200-500ms SLA) via local Ollama (Gemma 2 2B / E4B)
   or synthetic diagnostic fallback.
3. Benchmark suite with 10 test cases, traffic split metrics, P50/P95 latencies,
   and latency reduction reporting via `rich.table`.
"""

import sys
import os
import time
import json
import math
import re
import statistics
from typing import Optional, Dict, Any, List, Tuple

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from pydantic import BaseModel, Field

try:
    import requests
except ImportError:
    requests = None

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
except ImportError:
    Console = None
    Table = None
    Panel = None
    box = None


# Simulated motor ramp rates (1/s), matching panel.html: ~95% of a speed change within 2.5 s,
# e-stop spin-down within ~0.75 s
RAMP_RATE = 1.2
RAMP_RATE_ESTOP = 4.0

# Default safety limits (value >= limit trips that level). Temperature and pressure thresholds
# match the device panel ("hot" above 80 °C, red line at 5 bar).
SAFETY_LIMITS = {
    "motor_rpm": {"label": "Motor speed", "unit": "RPM", "warning": 2800, "danger": 3200},
    "temp_01_c": {"label": "Temperature (temp_01)", "unit": "°C", "warning": 80.0, "danger": 95.0},
    "pressure_valve_B_bar": {"label": "Pressure (pressure_valve_B)", "unit": "bar", "warning": 4.5, "danger": 5.0},
}


def assess_value(name: str, value: float) -> Dict[str, Any]:
    """Grade one reading against SAFETY_LIMITS: normal / warning / danger."""
    lim = SAFETY_LIMITS[name]
    status = "danger" if value >= lim["danger"] else "warning" if value >= lim["warning"] else "normal"
    return {"value": value, "unit": lim["unit"], "status": status,
            "warning_limit": lim["warning"], "danger_limit": lim["danger"]}


# ============================================================================
# Schemas & Data Models
# ============================================================================

class RouteResult(BaseModel):
    prompt: str
    tier_selected: str  # "Tier 1 (Small Brain)" or "Tier 2 (Big Brain)"
    latency_ms: float
    output_payload: Dict[str, Any]
    abstention_triggered: bool = False
    expected_tier: Optional[str] = None


class MotorSpeedCommand(BaseModel):
    tool: str = "set_motor_speed"
    rpm: int = Field(..., description="Target motor RPM")


class ReadSensorCommand(BaseModel):
    tool: str = "read_sensor"
    sensor_id: str = Field(..., description="Identifier for sensor")


class EmergencyStopCommand(BaseModel):
    tool: str = "emergency_stop"
    reason: str = "Triggered by emergency protocol"


# ============================================================================
# Tiered Router Core Implementation
# ============================================================================

class TieredRouter:
    def __init__(self, ollama_url: str = "http://127.0.0.1:11434", ollama_model: str = "gemma4:e4b"):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.ollama_available = self._check_ollama()

        # Simulated device state, updated by Tier 1 commands and summarized by Tier 2 reports
        self.device_state = {"motor_target_rpm": 0, "estop": False, "sensor_reads": {}, "ramp": None}
        self.event_log: List[Dict[str, Any]] = []

        # Initialize LiteRT-LM Engine for Tier 1 (FunctionGemma 270M)
        self.litert_model_path = r"C:\aiw\gemma-tiered\models\functiongemma-270m\mobile_actions_q8_ekv1024.litertlm"
        self.litert_engine = None
        self.litert_tools = []
        if os.path.exists(self.litert_model_path):
            try:
                os.environ["GLOG_minloglevel"] = "3"
                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
                import litert_lm
                litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
                self.litert_engine = litert_lm.Engine(self.litert_model_path, backend=litert_lm.Backend.CPU())

                def emergency_stop(reason: str = "Immediate reflex halt") -> str:
                    """Halts or stops all actuators and motors immediately.
                    
                    Args:
                        reason: Optional trigger reason for emergency halt.
                    """
                    return "Emergency stop executed"

                def set_motor_speed(rpm: int) -> str:
                    """Sets or adjusts primary motor rotational speed in RPM.
                    
                    Args:
                        rpm: Target rotations per minute as an integer.
                    """
                    return f"Speed set to {rpm} RPM"

                def read_sensor(sensor_id: str) -> str:
                    """Reads telemetry or status from a hardware sensor.
                    
                    Args:
                        sensor_id: Sensor identifier, e.g. temp_01 or pressure_valve_B.
                    """
                    return f"Sensor {sensor_id} read"

                self.litert_tools = [
                    litert_lm.tool_from_function(emergency_stop),
                    litert_lm.tool_from_function(set_motor_speed),
                    litert_lm.tool_from_function(read_sensor),
                ]
            except Exception:
                self.litert_engine = None
                self.litert_tools = []
        
    def tier_status(self) -> str:
        """One-line summary of which engine each tier will use."""
        if self.litert_engine and self.litert_tools:
            tier1 = "LiteRT-LM FunctionGemma 270M Q8 (loaded)"
        else:
            tier1 = "regex fallback (FunctionGemma not loaded)"
        if self.ollama_available:
            tier2 = f"Ollama {self.ollama_model} (online @ {self.ollama_url})"
        else:
            tier2 = "unavailable (Ollama offline; reports use tracked state)"
        return f"Tier 1: {tier1}  |  Tier 2: {tier2}"

    def _check_ollama(self) -> bool:
        if requests is None:
            return False
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=1.0)
            return r.status_code == 200
        except Exception:
            return False

    def _tier1_reflex(self, prompt: str) -> Dict[str, Any]:
        """
        Tier 1 Small Brain Reflex Engine.
        Executes FunctionGemma 270M via LiteRT-LM with strict physical tool binding,
        neural & deterministic abstention, and schema fallback.
        """
        prompt_lower = prompt.lower().strip()

        # Safety Check: If prompt involves reasoning, diagnostics, anomalies, or ambiguity -> ABSTAIN
        diagnostic_triggers = [
            "diagnose", "failure", "vibrating", "vibration", "hz", "overheating",
            "why", "analyze", "safe speed limit", "calculate", "unusual", "smoke", "smell"
        ]
        if any(trigger in prompt_lower for trigger in diagnostic_triggers):
            return {
                "escalate": True,
                "reason": "Input involves causal diagnosis or safety-critical ambiguity requiring Big Brain reasoning."
            }

        # Report requests -> Big Brain summarizes current device state
        if re.search(r"\b(report|status|summary|summari[sz]e|overview|sitrep)\b", prompt_lower):
            return {
                "escalate": True,
                "mode": "report",
                "reason": "Status report requested; Big Brain summarizes current device state."
            }

        # Questions about current device state -> Big Brain answers from telemetry (never actuates)
        is_question = "?" in prompt_lower or re.match(
            r"(what|what's|whats|how|is|are|was|does|did|has|can|tell me|show me)\b", prompt_lower)
        is_imperative = re.match(r"(set|adjust|change|increase|decrease|stop|halt|read|run|start)\b", prompt_lower)
        mentions_device = re.search(
            r"\b(motor|rpm|speed|temp\w*|pressure|sensor|actuator|e-?stop|running|steady|stable|state)\b", prompt_lower)
        if is_question and mentions_device and not is_imperative:
            return {
                "escalate": True,
                "mode": "question",
                "reason": "Operator question about device state; Big Brain answers from telemetry."
            }

        # 1. Neural Function Calling with LiteRT-LM (FunctionGemma 270M)
        if self.litert_engine and self.litert_tools:
            try:
                conv = self.litert_engine.create_conversation(
                    tools=self.litert_tools,
                    automatic_tool_calling=False
                )
                resp = conv.send_message(prompt)
                conv.close()

                if resp.tool_calls:
                    tc = resp.tool_calls[0]
                    tool_name = tc.name
                    args = tc.arguments or {}

                    if tool_name == "emergency_stop":
                        reason = args.get("reason", "Immediate reflex halt")
                        cmd = EmergencyStopCommand(reason=reason)
                        return {
                            "escalate": False,
                            "engine": "LiteRT-LM (FunctionGemma 270M Q8)",
                            "command": cmd.model_dump()
                        }
                    elif tool_name == "set_motor_speed":
                        rpm_val = int(args.get("rpm", 0))
                        cmd = MotorSpeedCommand(rpm=rpm_val)
                        return {
                            "escalate": False,
                            "engine": "LiteRT-LM (FunctionGemma 270M Q8)",
                            "command": cmd.model_dump()
                        }
                    elif tool_name == "read_sensor":
                        sensor_id = str(args.get("sensor_id") or args.get("sensor", ""))
                        cmd = ReadSensorCommand(sensor_id=sensor_id)
                        return {
                            "escalate": False,
                            "engine": "LiteRT-LM (FunctionGemma 270M Q8)",
                            "command": cmd.model_dump()
                        }
                elif resp.contents and any(w in resp.contents.lower() for w in ["diagnose", "not have a tool", "cannot assist", "sorry"]):
                    return {
                        "escalate": True,
                        "reason": f"FunctionGemma 270M Neural Abstention: {resp.contents.strip()}"
                    }
            except Exception:
                pass  # Fall through to schema validator

        # Fallback In-Process Schema Validator
        engine_name = "LiteRT-LM (FunctionGemma 270M Q8)" if self.litert_engine else "In-Process Schema Validator"

        # 1. Emergency Stop
        if any(k in prompt_lower for k in ["stop", "halt", "kill switch", "emergency"]):
            cmd = EmergencyStopCommand(reason="Immediate reflex halt")
            return {"escalate": False, "engine": engine_name, "command": cmd.model_dump()}

        # 2. Set Motor Speed (strict extraction)
        motor_match = re.search(r"(?:motor.*?|speed.*?|rpm.*?)(\d+)\s*(?:rpm)?", prompt_lower)
        if not motor_match:
            motor_match = re.search(r"(\d+)\s*rpm", prompt_lower)
        if motor_match and ("motor" in prompt_lower or "rpm" in prompt_lower or "speed" in prompt_lower):
            rpm_val = int(motor_match.group(1))
            cmd = MotorSpeedCommand(rpm=rpm_val)
            return {"escalate": False, "engine": engine_name, "command": cmd.model_dump()}

        # 3. Read Sensor
        sensor_match = re.search(r"sensor\s*([a-zA-Z0-9_\-]+)", prompt_lower)
        if sensor_match:
            sensor_id = sensor_match.group(1)
            cmd = ReadSensorCommand(sensor_id=sensor_id)
            return {"escalate": False, "engine": engine_name, "command": cmd.model_dump()}

        # Out-of-scope / Unknown -> Default Abstention Rule (never guess)
        return {
            "escalate": True,
            "reason": "Unrecognized reflex command syntax; strictly abstaining to prevent unintended device actuation."
        }

    def _ollama_chat(self, system: str, user: str, num_predict: int = 250,
                     timeout: float = 90.0) -> Tuple[Optional[str], Optional[str]]:
        """Single-turn Ollama chat call. Returns (text, None) on success or (None, error) on failure."""
        if not (self.ollama_available and requests is not None):
            return None, "Ollama offline"
        try:
            payload = {
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "stream": False,
                "think": False,
                "options": {"temperature": 0.1, "num_predict": num_predict}
            }
            resp = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=timeout)
            if resp.status_code != 200:
                return None, f"Ollama returned HTTP {resp.status_code}"
            msg = resp.json().get("message", {})
            text = msg.get("content", "").strip() or msg.get("thinking", "").strip()
            return (text, None) if text else (None, "Ollama returned an empty response")
        except requests.Timeout:
            return None, f"Ollama timed out after {timeout:.0f} s"
        except Exception as e:
            return None, f"Ollama request failed: {type(e).__name__}"

    def _motor_rpm_now(self) -> float:
        """Estimated actual motor speed: exponential ramp toward the setpoint (same model as the panel)."""
        ramp = self.device_state["ramp"]
        if ramp is None:
            return 0.0
        dt = time.monotonic() - ramp["t0"]
        return ramp["to"] + (ramp["from"] - ramp["to"]) * math.exp(-ramp["rate"] * dt)

    def _start_ramp(self, to_rpm: float, rate: float) -> None:
        self.device_state["ramp"] = {"from": self._motor_rpm_now(), "to": to_rpm, "rate": rate, "t0": time.monotonic()}

    def _apply_command(self, cmd: Dict[str, Any]) -> None:
        """Update the simulated device state from a Tier 1 command."""
        tool = cmd.get("tool")
        if tool == "emergency_stop":
            self.device_state["estop"] = True
            self._start_ramp(0, RAMP_RATE_ESTOP)
        elif tool == "set_motor_speed":
            # Commanding a speed clears the e-stop (demo behavior, mirrors the device panel)
            self.device_state["estop"] = False
            self.device_state["motor_target_rpm"] = max(0, int(cmd.get("rpm", 0)))
            self._start_ramp(self.device_state["motor_target_rpm"], RAMP_RATE)
        elif tool == "read_sensor":
            reads = self.device_state["sensor_reads"]
            sid = cmd.get("sensor_id", "")
            reads[sid] = reads.get(sid, 0) + 1

    def _status_snapshot(self) -> Dict[str, Any]:
        """Telemetry handed to the Big Brain for reports (same physics as the device panel)."""
        setpoint = self.device_state["motor_target_rpm"]
        estop = self.device_state["estop"]
        effective_setpoint = 0 if estop else setpoint
        current = self._motor_rpm_now()
        at_setpoint = abs(current - effective_setpoint) <= max(5.0, 0.02 * effective_setpoint)
        running = current > 5
        current_shown = round(current if not at_setpoint else effective_setpoint)
        temp = round(24 + current * 0.025, 1)
        pressure = round(1 + current * 0.0012, 2)
        return {
            "overall": "E-STOP ENGAGED" if estop else ("RUNNING" if running else "IDLE"),
            "emergency_stop_latched": estop,
            "motor": {
                "setpoint_rpm": setpoint,
                "estimated_current_rpm": current_shown,
                "at_setpoint": at_setpoint,
                "ramping": not at_setpoint,
                "running": running,
                "note": "estimated actual speed from a simulated ramp model; no encoder attached"
            },
            "actuator": "halted" if estop else ("cycling" if current > 30 else "idle"),
            "estimated_readings": {
                "temp_01_c": temp,
                "pressure_valve_B_bar": pressure,
                "note": "simulated estimates derived from motor speed; no live hardware attached"
            },
            "safety_assessment": self._assess_safety(current_shown, setpoint, temp, pressure, estop),
            "sensor_read_counts": dict(self.device_state["sensor_reads"]),
            "recent_activity": self.event_log[-10:],
            "commands_handled": {
                "tier1": sum(1 for e in self.event_log if e["tier"] == "T1"),
                "tier2": sum(1 for e in self.event_log if e["tier"] == "T2")
            }
        }

    @staticmethod
    def _assess_safety(rpm: float, setpoint: int, temp: float, pressure: float, estop: bool) -> Dict[str, Any]:
        """Grade current readings against SAFETY_LIMITS and produce a one-line verdict."""
        checks = {
            "motor_rpm": assess_value("motor_rpm", rpm),
            "temp_01_c": assess_value("temp_01_c", temp),
            "pressure_valve_B_bar": assess_value("pressure_valve_B_bar", pressure),
        }
        alerts = []
        for name, c in checks.items():
            if c["status"] != "normal":
                limit = c[f"{c['status']}_limit"]
                alerts.append({"level": c["status"], "message":
                               f"{SAFETY_LIMITS[name]['label']} {c['value']} {c['unit']} over "
                               f"{c['status']} limit ({limit} {c['unit']})"})
        setpoint_status = assess_value("motor_rpm", setpoint)["status"]
        if setpoint_status != "normal" and checks["motor_rpm"]["status"] == "normal" and not estop:
            alerts.append({"level": setpoint_status, "message":
                           f"Motor setpoint {setpoint} RPM over {setpoint_status} limit; "
                           "speed will exceed it once ramped"})
        if estop:
            alerts.append({"level": "warning", "message": "Emergency stop is latched; all motion halted"})

        levels = {a["level"] for a in alerts}
        overall = "DANGER" if "danger" in levels else "WARNING" if levels else "ALL VALUES NORMAL"
        if overall == "ALL VALUES NORMAL":
            summary = "All values normal"
        else:
            summary = f"{overall.title()}: " + "; ".join(a["message"] for a in alerts)
        return {"overall": overall, "summary": summary, "alerts": alerts, "checks": checks}

    @staticmethod
    def _brief_line(snap: Dict[str, Any]) -> str:
        motor = snap["motor"]
        est = snap["estimated_readings"]
        if motor["at_setpoint"]:
            ramp = ""
        elif snap["emergency_stop_latched"]:
            ramp = ", spinning down"
        else:
            ramp = f", ramping to {motor['setpoint_rpm']}"
        return (f"{snap['safety_assessment']['summary']}. Motor ~{motor['estimated_current_rpm']} RPM{ramp}, "
                f"temp_01 {est['temp_01_c']} °C, pressure_valve_B {est['pressure_valve_B_bar']} bar (simulated).")

    def _synthetic_report(self, snap: Dict[str, Any]) -> str:
        """Deterministic report built from the snapshot when Ollama is unavailable."""
        motor = snap["motor"]
        est = snap["estimated_readings"]
        safety = snap["safety_assessment"]
        reads = ", ".join(f"{k} x{v}" for k, v in snap["sensor_read_counts"].items()) or "none"
        recent = "; ".join(f"{e['prompt']} -> {e['result']}" for e in snap["recent_activity"][-5:]) or "none"

        recs = []
        for a in safety["alerts"]:
            if "Emergency stop" in a["message"]:
                recs.append("Investigate the e-stop cause before resuming; command a motor speed to clear it.")
            else:
                recs.append(f"{a['message']}; reduce motor speed.")
        if not recs:
            recs.append("None; all values within limits.")

        return (
            f"Safety: {safety['summary']}\n"
            f"1. Overall Status: {snap['overall']}\n"
            f"2. Motor & Actuators: running at ~{motor['estimated_current_rpm']} RPM "
            f"(setpoint {motor['setpoint_rpm']} RPM, {'at setpoint' if motor['at_setpoint'] else 'ramping'}); "
            f"actuator {snap['actuator']}; "
            f"e-stop {'LATCHED' if snap['emergency_stop_latched'] else 'clear'}.\n"
            f"3. Sensors: temp_01 ≈ {est['temp_01_c']} °C, pressure_valve_B ≈ {est['pressure_valve_B_bar']} bar "
            f"(simulated estimates). Reads so far: {reads}.\n"
            f"4. Recent Activity: {recent}\n"
            f"5. Recommendations: {' '.join(recs)}"
        )

    def _tier2_report(self, prompt: str) -> Dict[str, Any]:
        """Tier 2 report mode: summarize current device state for the operator."""
        snap = self._status_snapshot()
        brief = bool(re.search(r"\b(brief|short|quick|one[- ]line|tl;?dr)\b", prompt.lower()))
        verdict = snap["safety_assessment"]["summary"]
        if brief:
            system = (
                "You are a plant operations assistant. Write a 1-2 sentence status report from the telemetry JSON. "
                f'Your first words must be exactly: "{verdict}." Then give the key values: motor speed '
                "(mention the setpoint only if ramping), temperature and pressure, with units. "
                "Note once that values are simulated. No preamble, no field names."
            )
            # Brief reports only need the essentials; a smaller prompt is much faster on local E4B
            prompt_snap = {k: snap[k] for k in ("overall", "emergency_stop_latched", "motor", "estimated_readings")}
        else:
            system = (
                "You are a plant operations assistant. Write a concise operator status report from the telemetry JSON. "
                f'Your first line must be exactly: "Safety: {verdict}". Then use exactly these sections: '
                "1. Overall Status, 2. Motor & Actuators, 3. Sensors, 4. Recent Activity, 5. Recommendations. "
                "Judge normal vs abnormal only from safety_assessment, which applies the configured limits. "
                "In Recommendations, give one concrete action per alert (e.g. reduce speed below the warning "
                "limit), or write 'None; all values within limits.' if there are none. Use only facts present in "
                "the telemetry. No preamble."
            )
            prompt_snap = snap
        user = f"Request: {prompt}\n\nTelemetry:\n{json.dumps(prompt_snap, indent=2)}"
        text, err = self._ollama_chat(system, user, num_predict=100 if brief else 300)
        if text:
            return {"engine": f"Ollama ({self.ollama_model})", "mode": "report", "report": text, "telemetry": snap}
        fallback = self._brief_line(snap) if brief else self._synthetic_report(snap)
        return {
            "engine": "Tracked-State Report Builder (no model)",
            "mode": "report",
            "report": f"Big Brain unavailable ({err}); report built from tracked state.\n\n{fallback}",
            "telemetry": snap
        }

    def _tier2_question(self, prompt: str) -> Dict[str, Any]:
        """Tier 2 question mode: answer an operator's question about the current device state."""
        snap = self._status_snapshot()
        system = (
            "You are the plant's operations assistant. Answer the operator's question about the current device "
            "state using the telemetry JSON.\n"
            "- First sentence: the direct answer with the number and unit. For motor speed, quote "
            "motor.estimated_current_rpm as the current speed and say whether it is at the setpoint or still "
            "ramping toward motor.setpoint_rpm.\n"
            "- Then 1-3 short sentences of context: whether the state is steady, anything noteworthy in recent "
            "activity, and a next step only if one is warranted.\n"
            "- If the question is whether something is normal, safe or within limits, answer from "
            "safety_assessment (it applies the configured warning and danger limits).\n"
            "- All readings come from a simulation with no live hardware; mention that once, briefly.\n"
            "- Never invent data that is not in the telemetry. No preamble, no quotation marks around values."
        )
        user = f"Question: {prompt}\n\nTelemetry:\n{json.dumps(snap, indent=2)}"
        text, err = self._ollama_chat(system, user, num_predict=300)
        if text:
            return {"engine": f"Ollama ({self.ollama_model})", "mode": "question", "report": text, "telemetry": snap}
        return {
            "engine": "Tracked-State Report Builder (no model)",
            "mode": "question",
            "report": (f"Big Brain unavailable ({err}), so this question was not answered by a model. "
                       f"Current tracked state:\n\n{self._synthetic_report(snap)}"),
            "telemetry": snap
        }

    def _tier2_reflection(self, prompt: str) -> Dict[str, Any]:
        """
        Tier 2 Big Brain Reflection Engine.
        Calls local Ollama; if it is unavailable, says so rather than inventing a diagnosis.
        """
        text, err = self._ollama_chat(
            "You are an industrial robotics diagnostics expert. Provide a concise, structured diagnosis "
            "identifying: 1. Failure Mode, 2. Root Cause, 3. Safe Action & Speed Limit.",
            prompt
        )
        if text:
            return {
                "engine": f"Ollama ({self.ollama_model})",
                "diagnosis": text
            }

        return {
            "engine": "Big Brain unavailable",
            "diagnosis": (
                f"Big Brain unavailable ({err}); no diagnosis was produced. Tier 1 abstained, so no device "
                "action was taken. Retry once Ollama is responsive (check with `ollama ps`)."
            )
        }

    def _log_event(self, tier: str, prompt: str, result: str) -> None:
        self.event_log.append({"time": time.strftime("%H:%M:%S"), "tier": tier, "prompt": prompt, "result": result})
        del self.event_log[:-50]

    def route(self, prompt: str, expected_tier: Optional[str] = None) -> RouteResult:
        """Executes the tiered routing logic, tracks latency, and returns structured result."""
        t0 = time.perf_counter()
        
        # Tier 1 Reflex evaluation
        tier1_out = self._tier1_reflex(prompt)
        t_tier1 = (time.perf_counter() - t0) * 1000.0

        if not tier1_out.get("escalate", False):
            # Tier 1 handled successfully
            cmd = tier1_out.get("command", {})
            self._apply_command(cmd)
            self._log_event("T1", prompt, describe_command(cmd))
            return RouteResult(
                prompt=prompt,
                tier_selected="Tier 1 (Small Brain)",
                latency_ms=round(t_tier1, 2),
                output_payload=tier1_out,
                abstention_triggered=False,
                expected_tier=expected_tier
            )
        
        # Tier 1 Abstains -> Escalate to Tier 2 (report, question, or diagnosis)
        mode = tier1_out.get("mode")
        if mode == "report":
            tier2_out = self._tier2_report(prompt)
            self._log_event("T2", prompt, "status report")
        elif mode == "question":
            tier2_out = self._tier2_question(prompt)
            self._log_event("T2", prompt, "answered question")
        else:
            tier2_out = self._tier2_reflection(prompt)
            self._log_event("T2", prompt, "diagnosis")
        total_latency = (time.perf_counter() - t0) * 1000.0

        return RouteResult(
            prompt=prompt,
            tier_selected="Tier 2 (Big Brain)",
            latency_ms=round(total_latency, 2),
            output_payload={"abstention_reason": tier1_out.get("reason"), "reflection_output": tier2_out},
            abstention_triggered=True,
            expected_tier=expected_tier
        )


# ============================================================================
# Natural-Language Output Formatting
# ============================================================================

def describe_command(cmd: Dict[str, Any]) -> str:
    """Render a Tier 1 command dict as a plain-English sentence."""
    tool = cmd.get("tool")
    if tool == "emergency_stop":
        return f"Emergency stop issued: all actuators halted ({cmd.get('reason', 'no reason given')})."
    if tool == "set_motor_speed":
        return f"Setting motor speed to {cmd.get('rpm')} RPM."
    if tool == "read_sensor":
        return f"Reading sensor '{cmd.get('sensor_id')}'."
    return f"Unrecognized command: {tool}"


def format_result(res: RouteResult) -> str:
    """Render a RouteResult as polished natural-language text."""
    payload = res.output_payload
    if res.tier_selected == "Tier 1 (Small Brain)":
        engine = payload.get("engine", "Tier 1")
        return (
            f"{describe_command(payload.get('command', {}))}\n"
            f"  Handled by Tier 1 · {engine} · {res.latency_ms:.0f} ms"
        )

    out = payload.get("reflection_output", {})
    engine = out.get("engine", "Tier 2")
    mode = out.get("mode", "diagnosis")
    diagnosis = (out.get("report") or out.get("diagnosis") or "").strip() or f"No {mode} returned."
    return (
        f"Escalated to Tier 2 ({mode}) · {engine} · {res.latency_ms:.0f} ms\n"
        f"  Why: {payload.get('abstention_reason')}\n\n"
        f"{diagnosis}"
    )



# ============================================================================
# Benchmark Test Suite
# ============================================================================

TEST_SUITE = [
    # 6 Routine physical device commands (Expect Tier 1 < 20ms)
    {"prompt": "Set primary motor to 1400 RPM", "expected": "Tier 1 (Small Brain)"},
    {"prompt": "Stop all actuators immediately", "expected": "Tier 1 (Small Brain)"},
    {"prompt": "Read sensor temp_01", "expected": "Tier 1 (Small Brain)"},
    {"prompt": "Adjust motor speed to 2500 RPM", "expected": "Tier 1 (Small Brain)"},
    {"prompt": "Read sensor pressure_valve_B", "expected": "Tier 1 (Small Brain)"},
    {"prompt": "Emergency halt right now", "expected": "Tier 1 (Small Brain)"},

    # 4 Complex / Ambiguous Diagnostics (Expect Tier 2 via Abstention)
    {"prompt": "Motor 1 is vibrating at 120Hz while reading 65C; diagnose likely failure mode and compute safe speed limit", "expected": "Tier 2 (Big Brain)"},
    {"prompt": "Actuator speed is fluctuating wildly between 800 and 1500 RPM with unusual acoustic noise", "expected": "Tier 2 (Big Brain)"},
    {"prompt": "Sensor temp_01 reads 98C and smoke is detected near the gearbox; calculate thermal dissipation risk", "expected": "Tier 2 (Big Brain)"},
    {"prompt": "Why is the conveyor belt slipping whenever torque demand exceeds 45Nm?", "expected": "Tier 2 (Big Brain)"}
]


def run_benchmark(debug: bool = False, panel=None):
    console = Console() if Console else None
    
    if console:
        console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")
        console.print("[bold green]  TIERED ROUTING HARNESS: SMALL BRAIN / BIG BRAIN BENCHMARK  [/bold green]")
        console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]\n")
    else:
        print("=== TIERED ROUTING HARNESS BENCHMARK ===")

    router = TieredRouter()
    if console:
        console.print(f"[bold]{router.tier_status()}[/bold]\n")
    else:
        print(router.tier_status() + "\n")

    # Warmup
    router.route("Read sensor test_init")
    router.route("Diagnose motor failure")

    results: List[RouteResult] = []
    
    for idx, item in enumerate(TEST_SUITE, 1):
        prompt = item["prompt"]
        exp = item["expected"]
        res = router.route(prompt, expected_tier=exp)
        results.append(res)
        if panel:
            panel.publish(res)
        
        # Real-time feedback
        status_color = "green" if res.tier_selected == exp else "red"
        tier_tag = f"[{status_color}]{res.tier_selected}[/{status_color}]" if console else res.tier_selected
        if console:
            console.print(f"[bold]#{idx:02d}[/bold] {prompt[:55]:<55} -> {tier_tag} ({res.latency_ms} ms)")
            if res.tier_selected == "Tier 1 (Small Brain)":
                cmd = res.output_payload.get("command", {})
                cmd_text = cmd if debug else describe_command(cmd)
                console.print(f"     [dim]Reflex Command:[/dim] [cyan]{cmd_text}[/cyan]")
            else:
                diag = res.output_payload.get("reflection_output", {}).get("diagnosis", "")
                diag_lines = [l.strip() for l in diag.split("\n") if l.strip()]
                preview = " | ".join(diag_lines[:2])
                console.print(f"     [dim]Big Brain Output:[/dim] [yellow]{preview[:100]}...[/yellow]")
        else:
            print(f"#{idx:02d} {prompt[:55]:<55} -> {res.tier_selected} ({res.latency_ms} ms)")
            if res.tier_selected == "Tier 1 (Small Brain)":
                cmd = res.output_payload.get("command", {})
                print(f"     Reflex Command: {cmd if debug else describe_command(cmd)}")
            else:
                diag = res.output_payload.get("reflection_output", {}).get("diagnosis", "")
                diag_lines = [l.strip() for l in diag.split("\n") if l.strip()]
                preview = " | ".join(diag_lines[:2])
                print(f"     Big Brain Output: {preview[:100]}...")

    # Compute Statistics
    tier1_latencies = [r.latency_ms for r in results if r.tier_selected == "Tier 1 (Small Brain)"]
    tier2_latencies = [r.latency_ms for r in results if r.tier_selected == "Tier 2 (Big Brain)"]
    
    t1_count = len(tier1_latencies)
    t2_count = len(tier2_latencies)
    total_calls = len(results)

    t1_p50 = statistics.median(tier1_latencies) if tier1_latencies else 0.0
    t1_p95 = statistics.quantiles(tier1_latencies, n=20)[-1] if len(tier1_latencies) >= 2 else t1_p50

    t2_p50 = statistics.median(tier2_latencies) if tier2_latencies else 0.0
    t2_p95 = statistics.quantiles(tier2_latencies, n=20)[-1] if len(tier2_latencies) >= 2 else t2_p50

    # Baseline comparison: What if all 10 calls ran on Tier 2?
    baseline_avg_t2 = statistics.mean(tier2_latencies) if tier2_latencies else 280.0
    baseline_total_ms = baseline_avg_t2 * total_calls
    actual_total_ms = sum(r.latency_ms for r in results)
    latency_reduction_pct = (
        max(0.0, ((baseline_total_ms - actual_total_ms) / baseline_total_ms) * 100.0) if baseline_total_ms > 0 else 0.0
    )

    # Abstention verification rate (out of the 4 complex prompts, did all 4 abstain?)
    abstention_matches = sum(1 for r in results if (r.expected_tier == "Tier 2 (Big Brain)" and r.abstention_triggered))
    abstention_rate = (abstention_matches / 4.0) * 100.0

    # Display Rich Table
    if console and Table:
        table = Table(title="[bold yellow]Performance & Traffic Split Summary[/bold yellow]", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Measurement / Value", style="bold white")
        table.add_column("Target SLA / Requirement", style="dim")

        table.add_row("Total Evaluated Calls", str(total_calls), "10 test inputs")
        table.add_row("Traffic Split", f"{t1_count*10}% Tier 1 / {t2_count*10}% Tier 2", "Expected: 60% / 40%")
        table.add_row("Tier 1 P50 Latency", f"{t1_p50:.2f} ms", "< 20.0 ms SLA")
        table.add_row("Tier 1 P95 Latency", f"{t1_p95:.2f} ms", "< 20.0 ms SLA")
        table.add_row("Tier 2 P50 Latency", f"{t2_p50:.2f} ms", "200 - 500 ms SLA")
        table.add_row("Tier 2 P95 Latency", f"{t2_p95:.2f} ms", "200 - 500 ms SLA")
        table.add_row("Latency Reduction vs 100% Big Brain", f"[bold green]{latency_reduction_pct:.1f}% SAVED[/bold green]", "> 50% Reduction")
        table.add_row("Correct Abstention Rate", f"[bold green]{abstention_rate:.0f}%[/bold green]", "100% (Never Guess)")

        console.print("\n")
        console.print(table)
        console.print("\n[dim]Note: Tier 1 executes in-process reflex schema binding. Tier 2 escalates to local Ollama or synthetic engine.[/dim]\n")
    else:
        print("\n" + "=" * 50)
        print("SUMMARY METRICS")
        print("=" * 50)
        print(f"Total Calls: {total_calls}")
        print(f"Traffic Split: {t1_count*10}% Tier 1 / {t2_count*10}% Tier 2")
        print(f"Tier 1 P50 Latency: {t1_p50:.2f} ms")
        print(f"Tier 2 P50 Latency: {t2_p50:.2f} ms")
        print(f"Latency Reduction: {latency_reduction_pct:.1f}%")
        print(f"Abstention Accuracy: {abstention_rate:.0f}%")


def interactive_mode(debug: bool = False, panel=None):
    router = TieredRouter()
    print("\n--- Interactive Tiered Routing Console (Type 'exit' to quit) ---")
    print(router.tier_status())
    while True:
        try:
            prompt = input("\nEnter prompt > ").strip()
            if not prompt or prompt.lower() in ["exit", "quit"]:
                break
            res = router.route(prompt)
            if panel:
                panel.publish(res)
            if debug:
                print(f"Selected: {res.tier_selected} | Latency: {res.latency_ms} ms")
                print("Payload:", json.dumps(res.output_payload, indent=2))
            else:
                print(format_result(res))
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    debug = "--debug" in sys.argv
    panel = None
    if "--panel" in sys.argv:
        import webbrowser
        from device_panel import DevicePanel
        panel = DevicePanel()
        panel.start()
        print(f"Device panel: {panel.url}")
        webbrowser.open(panel.url)

    if "--benchmark" in sys.argv:
        run_benchmark(debug=debug, panel=panel)
        if panel:
            input("\nBenchmark done; panel still live. Press Enter to exit...")
    else:
        interactive_mode(debug=debug, panel=panel)
