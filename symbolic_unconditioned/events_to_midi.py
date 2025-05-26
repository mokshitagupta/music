from music21 import stream, note, duration, midi

def events_to_midi_polyphonic(events, output_file):
    s = stream.Stream()
    i = 0
    while i < len(events):
        tok = events[i]
        if 1000 <= tok < 2000:
            pitch = tok - 1000
            dur = 1.0
            if i+1 < len(events) and 2000 <= events[i+1] < 3000:
                dur = (events[i+1] - 2000) / 4
                i += 1
            s.append(note.Note(pitch, quarterLength=dur))
        elif 3000 <= tok < 4000:
            # Chord end marker, skip
            pass
        i += 1
    mf = midi.translate.streamToMidiFile(s)
    mf.open(output_file, 'wb')
    mf.write()
    mf.close()