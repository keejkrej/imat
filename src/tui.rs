use std::io::{self, Write};
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};

use crate::app::App;
use crate::render::render;

pub(crate) fn run_app(app: &mut App) -> Result<()> {
    enable_raw_mode().context("imat: failed to enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("imat: failed to switch to alternate screen")?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("imat: failed to initialize terminal")?;
    terminal
        .hide_cursor()
        .context("imat: failed to hide cursor")?;

    let result = run_event_loop(&mut terminal, app);

    disable_raw_mode().ok();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    let _ = terminal.show_cursor();
    let _ = terminal.backend_mut().flush();

    result
}

fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<()> {
    loop {
        #[cfg(feature = "sam")]
        warm_sam_embedding_if_needed(app)?;
        terminal.draw(|frame| render(frame, app))?;

        if !event::poll(Duration::from_millis(250))
            .context("imat: failed while waiting for input")?
        {
            continue;
        }

        match event::read().context("imat: failed to read input event")? {
            Event::Key(key) => {
                if app.on_key(key)? {
                    break;
                }
            }
            Event::Resize(_, _) => {}
            _ => {}
        }
    }

    Ok(())
}

#[cfg(feature = "sam")]
fn warm_sam_embedding_if_needed(app: &mut App) -> Result<()> {
    if !app.seg_mode {
        return Ok(());
    }
    if app.sam.is_none() {
        return Ok(());
    }
    let slice = app.rebuild_slice()?;
    let w = slice.width() as usize;
    let h = slice.height() as usize;
    if w == 0 || h == 0 {
        return Ok(());
    }
    let offset = app.current_offset_pixels()?;
    if let Err(error) = app.ensure_sam_embedding(slice.as_raw(), w, h, offset) {
        app.sam_last_error = Some(error.to_string().replace("imat: ", ""));
    }
    Ok(())
}
