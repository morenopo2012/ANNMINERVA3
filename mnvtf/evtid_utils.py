import numpy as np


def extract_id(evtid):
    if np.shape(evtid) == ():
        eventid = str(evtid)
    elif np.shape(evtid) == (1,):
        eventid = str(evtid[0])
    elif np.shape(evtid) == (1, 1):
        eventid = str(evtid[0][0])
    else:
        raise ValueError('Improper shape for event id!')
    return eventid


def decode_eventid(eventid):
    """ assume 64bit encoding """
    eventid = extract_id(eventid)
    phys_evt = eventid[-2:]
    eventid = eventid[:-2]
    gate = eventid[-4:]
    eventid = eventid[:-4]
    subrun = eventid[-4:]
    eventid = eventid[:-4]
    # note that run is not zero-padded
    run = eventid
    return (run, subrun, gate, phys_evt)


def decode_eventid32a(eventid):
    """ assume 32-bit "a" encoding """
    eventid = extract_id(eventid)
    phys_evt = eventid[-2:]
    eventid = eventid[:-2]
    # note that run is not zero-padded
    run = eventid
    return (run, phys_evt)


def decode_eventid32b(eventid):
    """ assume 32-bit "b" encoding """
    eventid = extract_id(eventid)
    gate = eventid[-4:]
    eventid = eventid[:-4]
    subrun = eventid.zfill(4)
    return (subrun, gate)


def decode_eventid32_combo(evta, evtb):
    (run, phys_evt) = decode_eventid32a(evta)
    (subrun, gate) = decode_eventid32b(evtb)
    return (run, subrun, gate, phys_evt)


def compare_evtid_encodings(evtid64, evtid32a, evtid32b):
    (run64, subrun64, gate64, phys_evt64) = decode_eventid(evtid64)
    (run32, subrun32, gate32, phys_evt32) = \
        decode_eventid32_combo(evtid32a, evtid32b)
    if run64 == run32 and \
       subrun64 == subrun32 \
       and gate64 == gate32 \
       and phys_evt64 == phys_evt32:
        return True
    return False


def encode_eventid(run, subrun, gate, phys_evt):
    run = '{0:06d}'.format(int(run))
    subrun = '{0:04d}'.format(int(subrun))
    gate = '{0:04d}'.format(int(gate))
    phys_evt = '{0:02d}'.format(int(phys_evt))
    return run + subrun + gate + phys_evt
