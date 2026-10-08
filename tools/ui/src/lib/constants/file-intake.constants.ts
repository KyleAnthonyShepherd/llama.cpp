// A file intake tool is a server tool that saves an attached file to disk and returns its path.
// The UI calls it on attach, it is never offered to the model.
export const FILE_INTAKE_TOOL_SUFFIX = '_put_file';

// same idea for a font file the server needs to draw a document as its author saw it
export const FONT_INTAKE_TOOL_SUFFIX = '_put_font';

// binary formats that go to the file intake tool, not to the text reader
export const FILE_INTAKE_EXTENSIONS = ['.docx', '.pptx'];

export const FONT_INTAKE_EXTENSIONS = ['.ttf', '.otf', '.ttc'];

export const FILE_INTAKE_MAX_BYTES = 32 * 1024 * 1024;

// the tool answers "saved: <absolute path>", then "needs-fonts: A | B" when the server lacks fonts
export const FILE_INTAKE_SAVED_PREFIX = 'saved: ';
export const FILE_INTAKE_NEEDS_FONTS_PREFIX = 'needs-fonts: ';
export const FILE_INTAKE_FONT_SEPARATOR = ' | ';

// first line of the note the model gets in place of the file content
export const FILE_INTAKE_NOTE_PREFIX = 'This file was saved to disk at: ';
