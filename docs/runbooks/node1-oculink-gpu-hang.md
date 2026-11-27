# Runbook: hung RX 9060 XT on the Node1 OCuLink dock

Applies to Node1 (RX 9060 XT gfx1200 on the OCuLink dock as Vulkan GPU0,
Radeon 780M gfx1103 as GPU1). Written after a Vulkan device-lost that left
the dock GPU unusable.

## How Sonder-Inference reports it

When a spawned `llamaserver` child logs a fatal accelerator fault, the
supervisor treats the device as unhealthy and stops. It does not relaunch
into the same device (a relaunch loop against a hung dock GPU can also wedge
the host). A fault is any of:

- `VK_ERROR_DEVICE_LOST` (Vulkan);
- `ggml_vulkan: device fault on <device>: <text>` (VK_EXT_device_fault output,
  llama.cpp #32008; present in the b11195 pin), including the following
  `address fault` / `vendor fault` lines;
- a CUDA/HIP error such as `unspecified launch failure`,
  `illegal memory access`, `device-side assert` or `launch timed out`.

What you see:

| Where | What |
| --- | --- |
| Runtime health, `backends[].runtime.diagnostics.device` | `status: "unhealthy"`, `backend`, `reason` (`device_lost`, `device_fault`, `fatal_error`), `device`, `description`, `details` (up to 8 fault lines) |
| Runtime warnings and `backend.warning` | `vulkan_runtime_error` (`reason: device_lost`) and/or `device_fault` |
| Telemetry event `backend.device_unhealthy` | `backend`, `reason`, `device`, `description` |
| Backend failure status | `llamaserver: device_unhealthy: ...`; requests fail fast with `unavailable` |

Detection needs the child log (`log_file`, or `--log-file` in `args`). Without
one the supervisor cannot see the fault and falls back to its ordinary
crash-restart budget.

The unhealthy state is terminal for that runtime process. Recover the device
(below), then restart the Sonder runtime (the `SonderInference` and
`SonderNodeRuntime` scheduled tasks on Node1).

## Recovery

A hung 9060 XT on the OCuLink dock is NOT recovered by:

- `pnputil /restart-device`;
- disabling and re-enabling the device in Device Manager or `pnputil`;
- a warm reboot of Node1.

After those the device stays at `CM_PROB_FAILED_POST_START` /
`FAILED_ADD`, and a warm reboot can also raise `DPC_WATCHDOG_VIOLATION` during
boot.

Only a hard power cycle of the dock recovers it:

1. Stop the Node1 services that use GPU0 (do not leave them restarting).
2. Shut Node1 down fully.
3. Remove power from the OCuLink dock (switch off or unplug its PSU), wait
   about 30 seconds, restore power.
4. Boot Node1; confirm the 9060 XT enumerates without a problem code
   (`Get-PnpDevice -Class Display`, status `OK`).
5. Start the services and check that `diagnostics.device` is absent and the
   backend is `ready`.

Do not run `test-backend-ops perf` on Node1: it has triggered this fault.

## Recommendation: TDR settings (not applied by this change)

Windows TDR (`HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`) decides
how long a stuck GPU job runs before the driver resets the device. This repo
changes no registry or driver settings; this is advice for the operator:

- A moderate `TdrDelay` / `TdrDdiDelay` increase can stop long but healthy
  compute kernels being killed. Keep it moderate: 60 seconds is too long,
  because a real hang then freezes the box for a minute.
- Never set `TdrLevel=0`. It disables recovery, so a hang is never reset and
  can only be cleared by the power cycle above.
- Whether a TDR adjustment helps a dock GPU that has already dropped off the
  bus is UNVERIFIED; the measured failure was a hard hang that TDR did not
  recover.
