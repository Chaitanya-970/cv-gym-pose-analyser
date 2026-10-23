"""Restrained light presentation for the local analyser; no behavior changes."""
CSS = '''
    :root {
      color-scheme: light;
      --bg: #f5f6f3; --card-bg: #ffffff; --card-border: #dce1db;
      --text: #252d28; --text-muted: #59665d;
      --primary: #31594a; --primary-hover: #244538;
      --primary-focus: #a8beb1;
      --success: #236341; --warning: #88560a; --danger: #b42318;
    }
    body { background: var(--bg); padding: 24px 24px 56px; font-size: 16px; }
    .container { max-width: 1040px; gap: 22px; }
    .review-nav { align-items: center; gap: 20px; padding: 0 0 20px; border-bottom: 1px solid var(--card-border); }
    .app-brand { color: var(--text); text-decoration: none; font-size: 16px; font-weight: 650; letter-spacing: -.3px; }
    .review-nav-link { background: transparent; border: 0; color: var(--text-muted); font-size: 13px; padding: 10px 0; }
    .review-nav-link:hover { background: transparent; color: var(--primary); text-decoration: underline; }
    header { text-align: left; padding: 20px 0 12px; margin: 0; }
    header h1 { color: var(--text); font-size: clamp(28px, 4vw, 38px); font-weight: 650; letter-spacing: -1.2px; line-height: 1.18; }
    header p { margin-top: 12px; font-size: 16px; color: var(--text-muted); max-width: 650px; }
    .card { padding: 26px; border-radius: 12px; box-shadow: none; }
    .card h2 { font-size: 19px; letter-spacing: -.3px; margin-bottom: 18px; }
    .card h3 { font-size: 17px; letter-spacing: -.2px; margin-bottom: 10px; }
    .card p { line-height: 1.65; }
    #upload-card { border-color: #cbd6cd; }
    .upload-form { gap: 18px; }
    .file-input-wrapper { padding: 30px 20px; min-height: 160px; background: #fafbf9; border: 1px dashed #7d8d7f; border-radius: 8px; gap: 10px; }
    .file-input-wrapper:hover, .file-input-wrapper.drag-over { background: #edf3ed; border-color: var(--primary); transform: none; }
    .file-label { font-size: 16px; color: var(--primary); }
    .selected-filename { color: var(--text-muted); font-size: 13px; }
    .btn { background: var(--primary); color: #ffffff; min-height: 46px; border-radius: 7px; padding: 11px 24px; box-shadow: none; }
    .btn:hover:not(:disabled) { background: var(--primary-hover); }
    .btn:disabled { background: #e5e9e3; color: #59665d; opacity: 1; }
    .btn-secondary { background: #eef1ec; border: 1px solid var(--card-border); color: var(--text); border-radius: 7px; }
    .btn-secondary:hover { background: #e2e8df; }
    .btn-seek { background: #f7f9f5; border-color: #b7c8ba; color: var(--primary); border-radius: 6px; }
    .btn-seek:hover { background: var(--primary); color: #fff; }
    .notice-box { background: #f7f9f5; border-left: 3px solid #9caf9d; border-radius: 6px; color: var(--text-muted); }
    .recording-guidance { order: 0; }
    .recording-guidance summary { cursor: pointer; min-height: 44px; display: list-item; list-style-position: inside; font-weight: 600; padding: 2px 0; }
    .recording-guidance[open] summary { margin-bottom: 16px; }
    .recording-guidance h2 { font-size: 16px; }
    .metrics-grid { gap: 14px; }
    .metric-card { background: #f7f9f5; border-radius: 8px; padding: 20px 14px; text-align: left; }
    .metric-value { color: var(--text); font-size: 36px; letter-spacing: -1px; font-variant-numeric: tabular-nums; }
    .metric-label { font-size: 12px; letter-spacing: .5px; }
    .data-table { font-size: 14px; }
    .data-table th { background: #edf1e9; color: #46544a; }
    .data-table td { background: #fff; }
    .data-table tr:nth-child(even) td { background: #fafbf9; }
    .media-box { background: #f7f9f5; border-radius: 8px; }
    .status-badge { border-radius: 5px; }
    .status-queued { background: #e8eef6; color: #25486a; }
    .status-running, .status-incomplete { background: #fbf0d5; color: #70520c; border: 1px solid #ebd9aa; animation: none; }
    .status-completed { background: #e5f0e5; color: #245534; }
    .status-failed, .status-interrupted { background: #fbe9e6; color: #9a2c23; }
    .status-unavailable, .status-unvalidated { background: #edf0eb; color: #4b594e; border-color: #cbd4c9; }
    .alert-danger { background: #fdf0ed; border-color: #e9bcb4; color: #912c20; }
    .alert-warning { background: #fcf5e5; border-color: #e4d1a5; color: #74500d; }
    input[type=number], select { font: inherit; color: var(--text); background: white; border: 1px solid #7d8d7f; border-radius: 5px; min-height: 44px; padding: 6px 8px; max-width: 100%; }
    input[type=checkbox] { accent-color: var(--primary); width: 17px; height: 17px; margin: 8px 6px 8px 0; vertical-align: middle; }
    form > label { line-height: 1.6; }
    a { color: var(--primary); }
    *:focus-visible { outline: 3px solid var(--primary); outline-offset: 3px; }
    .btn:focus-visible, .review-nav-link:focus-visible, input:focus-visible, a:focus-visible, button:focus-visible { box-shadow: none; }
    @media (max-width: 640px) {
      body { padding: 16px 14px 36px; }
      .container { gap: 16px; }
      .card { padding: 20px 16px; }
      .review-nav { align-items: flex-start; gap: 8px; }
      .review-nav-link { width: auto; max-width: 54%; text-align: right; line-height: 1.45; }
      header { padding: 12px 0 8px; }
      header p { font-size: 15px; }
      .metrics-grid { grid-template-columns: 1fr 1fr; }
      .file-input-wrapper { min-height: 150px; padding: 22px 12px; }
      .btn { width: 100%; }
    }
    #teng-events { display: grid; gap: 18px; margin: 14px 0; }
    #teng-events h4 { margin-bottom: 8px; font-size: 15px; }
    .teng-suggestion-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr)); gap: 10px; }
    .teng-suggestion { padding: 12px; border: 1px solid var(--card-border); border-radius: 7px; background: #fafbf9; }
    .teng-suggestion .btn-seek { width: 100%; }
    .teng-suggestion p { font-size: 13px; color: var(--text-muted); margin-top: 7px; }
    .teng-suggestion .btn-seek[aria-pressed=true] { background: #e3ece3; color: #244538; border-color: #31594a; font-weight: 600; }
    .advanced-reviews > summary, #teng-review-card summary { cursor: pointer; min-height: 44px; padding: 10px 0; font-weight: 600; }
    #video-review-context[hidden] { display: none; }
    #video-review-context { margin-top: 10px; }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { animation: none !important; transition: none !important; scroll-behavior: auto !important; }
    }
'''


def apply_simple_theme(html):
    # Cover both static styles and generated inline coaching styles.
    replacements = {
        '#0f172a': '#f5f6f3', '#1e293b': '#ffffff', '#334155': '#dce1db',
        '#475569': '#e8ece8', '#172033': '#fafbf9', '#f8fafc': '#252d28',
        '#94a3b8': '#59665d', '#38bdf8': '#31594a', '#0ea5e9': '#244538',
        'rgba(56, 189, 248, 0.4)': '#a8beb1',
        'rgba(56, 189, 248, 0.35)': '#b5c7b8',
        'rgba(56, 189, 248, 0.04)': '#f1f5ef',
        'rgba(56, 189, 248, 0.12)': '#edf3ed',
        'rgba(56, 189, 248, 0.08)': '#edf3ed',
    }
    for old,new in replacements.items():
        html=html.replace(old,new)
    html=html.replace('</style>', CSS+'\n  </style>',1)
    html=html.replace('justify-content: flex-end; margin-bottom: 8px;', 'justify-content: space-between; margin-bottom: 0;')
    html=html.replace('<nav class="review-nav"', '<nav aria-label="Application" class="review-nav"',1)
    html=html.replace('<a href="/review/" class="review-nav-link"', '<span class="app-brand">Squat Analyser</span><a href="/review/" class="review-nav-link"',1)
    html=html.replace('&rarr; Human Review Workspace (Label raw video BEFORE viewing model output)', 'Human Review Workspace')
    html=html.replace('Single-user local squat repetition counter and 2D kinematic feedback', 'Upload a squat video. Review your repetitions, movement and posture suggestions.')
    # Keep detailed recording guidance available without placing it before the main action.
    start=html.index('    <!-- Recording Guidance & Technical Notices -->')
    end=html.index('    <!-- Active Job Status Panel -->',start)
    section=html[start:end];split=section.index('    <!-- Upload Form -->')
    guidance,upload=section[:split],section[split:]
    guidance=guidance.replace('<div class="card">','<details class="card recording-guidance">\n      <summary>Recording tips &amp; limitations</summary>',1)
    close=guidance.rfind('</div>');guidance=guidance[:close]+'</details>'+guidance[close+6:]
    html=html[:start]+upload+guidance+html[end:]
    return html
