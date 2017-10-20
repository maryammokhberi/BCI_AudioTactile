# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 16:18:44 2017

@author: Amarantine
"""


def _read_vmrk_events(fname, event_id=None, response_trig_shift=0):
    """Read events from a vmrk file.

    Parameters
    ----------
    fname : str
        vmrk file to be read.
    event_id : dict | None
        The id of special events to consider in addition to those that
        follow the normal Brainvision trigger format ('S###').
        If dict, the keys will be mapped to trigger values on the stimulus
        channel. Example: {'SyncStatus': 1; 'Pulse Artifact': 3}. If None
        or an empty dict (default), only stimulus events are added to the
        stimulus channel. Keys are case sensitive.
    response_trig_shift : int | None
        Integer to shift response triggers by. None ignores response triggers.

    Returns
    -------
    events : array, shape (n_events, 3)
        An array containing the whole recording's events, each row representing
        an event as (onset, duration, trigger) sequence.
    """
    if event_id is None:
        event_id = dict()
    # read vmrk file
    with open(fname, 'rb') as fid:
        txt = fid.read()

    # we don't actually need to know the coding for the header line.
    # the characters in it all belong to ASCII and are thus the
    # same in Latin-1 and UTF-8
    header = txt.decode('ascii', 'ignore').split('\n')[0].strip()
    _check_mrk_version(header)
    if (response_trig_shift is not None and
            not isinstance(response_trig_shift, int)):
        raise TypeError("response_trig_shift must be an integer or None")

    # although the markers themselves are guaranteed to be ASCII (they
    # consist of numbers and a few reserved words), we should still
    # decode the file properly here because other (currently unused)
    # blocks, such as that the filename are specifying are not
    # guaranteed to be ASCII.

    codepage = 'utf-8'
    try:
        # if there is an explicit codepage set, use it
        # we pretend like it's ascii when searching for the codepage
        cp_setting = re.search('Codepage=(.+)',
                               txt.decode('ascii', 'ignore'),
                               re.IGNORECASE & re.MULTILINE)
        if cp_setting:
            codepage = cp_setting.group(1).strip()
        txt = txt.decode(codepage)
    except UnicodeDecodeError:
        # if UTF-8 (new standard) or explicit codepage setting fails,
        # fallback to Latin-1, which is Windows default and implicit
        # standard in older recordings
        txt = txt.decode('latin-1')

    # extract Marker Infos block
    m = re.search("\[Marker Infos\]", txt)
    if not m:
        return np.zeros(0)
    mk_txt = txt[m.end():]
    m = re.search("\[.*\]", mk_txt)
    if m:
        mk_txt = mk_txt[:m.start()]

    # extract event information
    items = re.findall("^Mk\d+=(.*)", mk_txt, re.MULTILINE)
    events, dropped = list(), list()
    for info in items:
        mtype, mdesc, onset, duration = info.split(',')[:4]
        onset = int(onset)
        duration = (int(duration) if duration.isdigit() else 1)
        if mdesc in event_id:
            trigger = event_id[mdesc]
        else:
            try:
                trigger = int(re.findall('[A-Za-z]*\s*?(\d+)', mdesc)[0])
            except IndexError:
                trigger = None
            if mtype.lower().startswith('response'):
                if response_trig_shift is not None:
                    trigger += response_trig_shift
                else:
                    trigger = None
        if trigger:
            events.append((onset, duration, trigger))
        else:
            if len(mdesc) > 0:
                dropped.append(mdesc)

    if len(dropped) > 0:
        dropped = list(set(dropped))
        examples = ", ".join(dropped[:5])
        if len(dropped) > 5:
            examples += ", ..."
        warn("Currently, {0} trigger(s) will be dropped, such as [{1}]. "
             "Consider using ``event_id`` to parse triggers that "
             "do not follow the 'S###' pattern.".format(
                 len(dropped), examples))

    events = np.array(events).reshape(-1, 3)
    np.save("exported_events.npy", events) #Maryam
    return events