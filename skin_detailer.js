/**
 * In-node UI for "Skin Detailer (SAM3 - Texture - Refine)".
 *
 * Adds a header band above the widgets that shows which mode the node is in,
 * what that mode will actually run, and -- in diffusion mode -- whether the
 * optional MODEL/CLIP/VAE inputs it needs are actually connected. Clicking the
 * band switches mode. Diffusion-only widgets are hidden while in texture mode.
 *
 * Implementation notes (verified against frontend 1.47/1.48):
 *  - `widget.hidden = true` is the supported hide. Hidden widgets still
 *    serialise, so this creates no holes in `widgets_values`.
 *  - Canvas space is reserved by wrapping `_arrangeWidgets` and `computeSize`,
 *    never by inserting widgets: a `serialize:false` widget in the middle of
 *    the list shifts every later value on reload.
 *  - `onDrawForeground` runs BEFORE positions are measured, so the reserved
 *    band's Y is unknown on the very first frame. The fix-up `arrange()` must
 *    happen in `onAdded`, NOT `onNodeCreated`: during
 *    `LGraph.configure -> createNode -> onNodeCreated` the node is not attached
 *    to a graph yet, and the base `_arrangeWidgets` dereferences the graph and
 *    throws NullGraphError -- which aborts the entire workflow load.
 *  - Nothing in here may throw during graph configure, so every entry point is
 *    wrapped. A cosmetic extension must never be able to break loading.
 *  - When Vue Nodes are enabled, canvas draw hooks do not run at all. The band
 *    simply does not appear; every widget still works, and mode is still
 *    switchable from the `mode` combo itself.
 */

import { app } from "../../scripts/app.js";

const NODE_ID = "SkinDetailer";

const MODE_TEXTURE = "skin texture (stable)";
const MODE_DIFFUSION = "texture + diffusion (experimental)";

/** Widgets that only apply in diffusion mode. Must match DIFFUSION_WIDGETS in node.py. */
const DIFFUSION_WIDGETS = [
  "seed", "control_after_generate", "steps", "cfg", "denoise", "preserve_tone",
  "skin_prompt", "sampler_name", "scheduler", "processing_mode",
];

/** Optional inputs diffusion mode depends on. */
const DIFFUSION_INPUTS = ["model", "clip", "vae"];

const BAND_H = 62;
const PAD = 10;

const THEME = {
  texture: {
    pill: "#1f6f4a",
    pillText: "#d8ffe9",
    accent: "#39c07f",
    label: "SKIN TEXTURE",
    sub: "stable",
    line1: "Real skin reference detail; original colour and size retained.",
    line2: "Skin detail amount adds texture; detail size controls its scale.",
  },
  diffusion: {
    pill: "#7a4a12",
    pillText: "#ffe9c9",
    accent: "#e0912f",
    label: "TEXTURE + DIFFUSION",
    sub: "experimental",
    line1: "Adds a masked low-denoise pass with your MODEL / CLIP / VAE.",
    line2: "May invent blemishes. Keep denoise low and preserve_tone high.",
  },
};

function roundRect(ctx, x, y, w, h, r) {
  const rad = Math.min(r, h / 2, w / 2);
  ctx.beginPath();
  ctx.moveTo(x + rad, y);
  ctx.arcTo(x + w, y, x + w, y + h, rad);
  ctx.arcTo(x + w, y + h, x, y + h, rad);
  ctx.arcTo(x, y + h, x, y, rad);
  ctx.arcTo(x, y, x + w, y, rad);
  ctx.closePath();
}

function getWidget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function isDiffusion(node) {
  return getWidget(node, "mode")?.value === MODE_DIFFUSION;
}

/** True when the named optional input exists and has a link. */
function inputConnected(node, name) {
  const slot = node.inputs?.find((i) => i.name === name);
  return !!(slot && slot.link != null);
}

/** Diffusion mode accepts CLIP *or* a positive conditioning. */
function inputSatisfied(node, name) {
  if (name === "clip") {
    return inputConnected(node, "clip") || inputConnected(node, "positive");
  }
  return inputConnected(node, name);
}

function applyMode(node) {
  const diffusion = isDiffusion(node);
  // Keep the historical slot for workflow serialization, but never offer
  // recursive refinement or let a saved six-pass value re-enable it.
  const legacyPasses = getWidget(node, "texture_passes");
  if (legacyPasses) {
    legacyPasses.value = 1;
    legacyPasses.hidden = true;
  }
  const referenceAmount = getWidget(node, "skin_detail_amount");
  if (referenceAmount) referenceAmount.hidden = diffusion;
  for (const name of DIFFUSION_WIDGETS) {
    const w = getWidget(node, name);
    if (w) w.hidden = !diffusion;
  }

  const theme = diffusion ? THEME.diffusion : THEME.texture;
  node.bgcolor = diffusion ? "#3a2f1c" : "#20302a";
  node.color = diffusion ? "#5a4520" : "#2c4a3c";
  node._skinDetailerTheme = theme;

  // Widget visibility changes the required height.
  const size = node.computeSize();
  node.setSize([Math.max(node.size[0], size[0]), Math.max(size[1], BAND_H + 40)]);
  // setDirtyCanvas touches the graph; during configure there isn't one yet.
  if (node.graph) node.setDirtyCanvas(true, true);
}

/** applyMode, but incapable of aborting a workflow load. */
function safeApplyMode(node) {
  try {
    applyMode(node);
  } catch (err) {
    console.error("[SkinDetailer] applyMode failed", err);
  }
}

/**
 * Re-measure widget positions. Only ever safe once the node is in a graph:
 * the base _arrangeWidgets throws NullGraphError otherwise.
 */
function safeArrange(node) {
  if (!node.graph) return;
  try {
    node.arrange?.();
  } catch (err) {
    console.error("[SkinDetailer] arrange failed", err);
  }
}

function drawBand(node, ctx) {
  if (node.flags?.collapsed) return;
  const top = node._skinDetailerBandY;
  if (top == null) return;

  const theme = node._skinDetailerTheme ||
    (isDiffusion(node) ? THEME.diffusion : THEME.texture);
  const width = node.size[0];

  ctx.save();

  // Band background + left accent rail.
  roundRect(ctx, PAD * 0.6, top, width - PAD * 1.2, BAND_H - 8, 6);
  ctx.fillStyle = "rgba(0,0,0,0.22)";
  ctx.fill();
  ctx.fillStyle = theme.accent;
  roundRect(ctx, PAD * 0.6, top, 3, BAND_H - 8, 2);
  ctx.fill();

  // Mode pill.
  ctx.font = "bold 10px sans-serif";
  const labelW = ctx.measureText(theme.label).width;
  const pillW = labelW + 16;
  const pillX = PAD + 4;
  const pillY = top + 7;
  roundRect(ctx, pillX, pillY, pillW, 16, 8);
  ctx.fillStyle = theme.pill;
  ctx.fill();
  ctx.fillStyle = theme.pillText;
  ctx.textBaseline = "middle";
  ctx.fillText(theme.label, pillX + 8, pillY + 8);

  // "stable" / "experimental" qualifier.
  ctx.font = "italic 10px sans-serif";
  ctx.fillStyle = theme.accent;
  ctx.fillText(theme.sub, pillX + pillW + 7, pillY + 8);

  // Hint on the right.
  ctx.font = "9px sans-serif";
  ctx.fillStyle = "rgba(255,255,255,0.38)";
  ctx.textAlign = "right";
  ctx.fillText("click to switch", width - PAD - 4, pillY + 8);
  ctx.textAlign = "left";

  // Explanation lines.
  ctx.font = "10px sans-serif";
  ctx.fillStyle = "rgba(255,255,255,0.68)";
  ctx.fillText(theme.line1, pillX, top + 32);

  if (isDiffusion(node)) {
    // Live readiness of the inputs this mode needs.
    let x = pillX;
    ctx.font = "10px sans-serif";
    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.fillText("needs:", x, top + 48);
    x += ctx.measureText("needs:").width + 7;
    for (const name of DIFFUSION_INPUTS) {
      const ok = inputSatisfied(node, name);
      ctx.beginPath();
      ctx.arc(x + 3, top + 48, 3, 0, Math.PI * 2);
      ctx.fillStyle = ok ? "#39c07f" : "#d75d5d";
      ctx.fill();
      x += 10;
      ctx.fillStyle = ok ? "rgba(255,255,255,0.72)" : "#d75d5d";
      const text = name.toUpperCase();
      ctx.fillText(text, x, top + 48);
      x += ctx.measureText(text).width + 11;
    }
  } else {
    ctx.fillStyle = "rgba(255,255,255,0.45)";
    const referenceEnabled = Number(getWidget(node, "skin_detail_amount")?.value ?? 0.65) > 0;
    const bypassed = Number(getWidget(node, "texture_strength")?.value) === 0 ||
      (getWidget(node, "texture_model")?.value === "off" && !referenceEnabled);
    const hint = bypassed ? "Texture is off; the image passes through unchanged." :
      referenceEnabled ? theme.line2 : "Model only. Raise skin detail amount to supply missing texture.";
    ctx.fillText(hint, pillX, top + 48);
  }

  ctx.restore();
}

app.registerExtension({
  name: "comfyui-skin-detailer.ui",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);

      const modeWidget = getWidget(this, "mode");
      if (modeWidget) {
        const prev = modeWidget.callback;
        modeWidget.callback = (...args) => {
          const out = prev?.apply(this, args);
          safeApplyMode(this);
          return out;
        };
      }

      // Reserve the band's height above the widget stack.
      const baseArrange = this._arrangeWidgets;
      if (baseArrange) {
        this._arrangeWidgets = function (startY, ...rest) {
          this._skinDetailerBandY = startY + 4;
          // The base implementation dereferences the graph; refusing to call it
          // is strictly better than throwing out of a workflow load.
          if (!this.graph) return undefined;
          return baseArrange.call(this, startY + BAND_H, ...rest);
        };
      }
      const baseComputeSize = this.computeSize;
      this.computeSize = function (...args) {
        const size = baseComputeSize.apply(this, args);
        size[1] += BAND_H;
        size[0] = Math.max(size[0], 300);
        return size;
      };

      // NOTE: no arrange() here. The node has no graph during configure, and
      // the base _arrangeWidgets would throw and abort the workflow load.
      safeApplyMode(this);
      return r;
    };

    // Fires once the node is actually attached to a graph -- the earliest
    // point at which arrange() is legal.
    const onAdded = nodeType.prototype.onAdded;
    nodeType.prototype.onAdded = function () {
      const r = onAdded?.apply(this, arguments);
      safeApplyMode(this);
      safeArrange(this);
      return r;
    };

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      const r = onDrawForeground?.apply(this, arguments);
      try {
        drawBand(this, ctx);
      } catch (err) {
        console.error("[SkinDetailer] band draw failed", err);
      }
      return r;
    };

    // Clicking the band toggles mode. onMouseDown only fires when no widget is
    // under the cursor, which is exactly the reserved band.
    const onMouseDown = nodeType.prototype.onMouseDown;
    nodeType.prototype.onMouseDown = function (event, pos) {
      const top = this._skinDetailerBandY;
      if (top != null && pos[1] >= top && pos[1] <= top + BAND_H - 8) {
        const w = getWidget(this, "mode");
        if (w) {
          w.value = w.value === MODE_DIFFUSION ? MODE_TEXTURE : MODE_DIFFUSION;
          w.callback?.(w.value, app.canvas, this, pos, event);
          safeApplyMode(this);
          return true; // consume
        }
      }
      return onMouseDown?.apply(this, arguments);
    };

    // Widget visibility and colours must be restored after a workflow load.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      const texture = getWidget(this, "texture_model");
      if (typeof texture?.value === "string" && texture.value.startsWith("auto")) {
        texture.value = "auto (ClearReality photo detail)";
      }
      safeApplyMode(this);
      safeArrange(this); // no-ops until the node is in a graph
      return r;
    };

    // Connecting or unplugging MODEL/CLIP/VAE changes the readiness dots.
    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const r = onConnectionsChange?.apply(this, arguments);
      this.setDirtyCanvas(true, false);
      return r;
    };
  },
});
