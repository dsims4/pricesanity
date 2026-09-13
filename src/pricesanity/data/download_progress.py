"""Compact synchronous progress, with redraws only on capable terminals."""

import os
import shutil
import sys


def component_progress(component_state: dict) -> dict:
    """Count validated chunks rather than the position of the active request.

    Args:
        component_state: Saved state and planned chunks for one component.

    Returns:
        Completed and total chunk counts, percentage, and committed row count.
    """

    # Later chunks may already be complete after a resume; the current request's
    # position therefore cannot represent the amount of validated data on disk.
    completed_chunk_states = [
        chunk for chunk in component_state['chunks'] if chunk['state'] == 'complete'
    ]

    # Count the same planned partition in both values so a skipped chunk still
    # contributes to the denominator without being counted as completed work.
    completed_chunks = len(completed_chunk_states)
    total_chunks = len(component_state['chunks'])

    # An empty partition has no outstanding work. Sum only committed rows so a
    # failed response cannot inflate the progress display.
    return {
        'completed_chunks': completed_chunks,
        'total_chunks': total_chunks,
        'percent': 100 * completed_chunks / total_chunks if total_chunks else 100,
        'rows': sum(chunk.get('stats', {}).get('rows', 0) for chunk in completed_chunk_states),
    }


class DownloadProgress:
    """Keep one interactive display; redirected output gets only useful events."""

    def __init__(self, stream=None):
        """Choose terminal capabilities once for this operation's display.

        Args:
            stream: Output destination; omitted streams use standard output.
        """

        # Resolve standard output at construction so redirection and test streams
        # are respected instead of retaining an earlier process-wide reference.
        self.stream = sys.stdout if stream is None else stream

        # Cursor controls require both a terminal stream and a capable terminal;
        # redirected logs and dumb terminals need ordinary text.
        self.interactive = self.stream.isatty() and os.environ.get('TERM') != 'dumb'

        # Remember display height for redraws and the component for log throttling.
        self.lines = 0
        self.last_component = None

    def render(self, manifest: dict, context: str, warnings: int, errors: int,
               *, retry: int = 0, final: bool = False, event: bool = False) -> None:
        """Show cumulative progress without flooding redirected logs.

        Args:
            manifest: Validated state used to calculate component completion.
            context: Active component, range, and operation description.
            warnings: Cumulative warning count, including previous invocations.
            errors: Cumulative error count, including previous invocations.
            retry: Retry number for the active chunk.
            final: Whether to include the remaining gaps before the operation ends.
            event: Whether this update must appear even in redirected output.
        """

        # Range changes share the same component, allowing redirected output to
        # suppress routine requests while still reporting component changes.
        component = context.split(' ', 1)[0]

        # Interactive terminals can redraw each update; logs need significant
        # events, retries, and the final state rather than repeated dashboards.
        if not (self.interactive or final or event or component != self.last_component):
            return

        # Build one complete display before writing so all component counts refer
        # to the same checkpoint view.
        self.last_component = component
        lines = [f"Current: {context} | Retry: {retry}"]

        # Include inactive components because a resumed request may already have
        # useful work committed beyond the range currently being downloaded.
        for name, component_state in manifest['components'].items():
            progress_counts = component_progress(component_state)
            lines.append(f"{name.capitalize()}: {component_state['state'].upper()} | "
                         f"{progress_counts['completed_chunks']}/"
                         f"{progress_counts['total_chunks']} chunks | "
                         f"{progress_counts['percent']:.0f}% | {progress_counts['rows']:,} rows")

            # This boundary stops at the first gap, unlike the cumulative counts
            # above, which also include completed chunks after a missing range.
            committed_boundary = component_state['last_successful_exclusive_boundary']
            lines.append(f"  Committed through: {committed_boundary or 'none'} exclusive")

            # Remaining ranges matter when the operation stops, without adding
            # the same incomplete-range list to every intermediate update.
            if final:
                missing = [
                    f"{chunk['start']}..{chunk['end']}"
                    for chunk in component_state['chunks'] if chunk['state'] != 'complete'
                ]

                # Limit long requests to three examples while still reporting
                # how many additional ranges remain unresolved.
                if missing:
                    suffix = f" (+{len(missing)-3} more)" if len(missing) > 3 else ''
                    lines.append(f"  Missing [start,end): {', '.join(missing[:3])}{suffix}")

        # Keep diagnostic totals visible while full messages remain in the log.
        lines.append(f"Warnings: {warnings} | Errors: {errors}")

        # Redraw only when cursor movement is supported by the selected output.
        if self.interactive:
            # Limit each line to the terminal width so wrapping cannot leave stale
            # dashboard lines behind on the next redraw.
            width = max(1, shutil.get_terminal_size().columns - 1)

            # Return to the previous display's first line before replacing it.
            if self.lines:
                self.stream.write(f'\x1b[{self.lines}A')

            # Clear old text so a shorter replacement leaves no stale characters.
            for line in lines:
                self.stream.write('\x1b[2K' + line[:width] + '\n')

            # The final summary can have more lines than an intermediate update.
            self.lines = len(lines)
        else:
            # Plain text remains readable when saved to a file or piped elsewhere.
            self.stream.write('\n'.join(lines) + '\n')

        # Flush now so progress remains visible while the next vendor call blocks.
        self.stream.flush()
