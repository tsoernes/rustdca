use super::Opt;
use gridfuncs::*;
use ordered_float::*;
use rand::distributions::{Distribution, Exp, Uniform};
use rand::thread_rng;
use revord::RevOrd;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::fmt;

#[derive(PartialEq, Eq, Ord, PartialOrd, Clone, Debug)]
pub enum EType {
    NEW = 0,
    END = 1,
    HOFF = 2,
}

impl fmt::Display for EType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            EType::NEW => write!(f, "NEW"),
            EType::END => write!(f, "END"),
            EType::HOFF => write!(f, "HOFF"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Event {
    pub id: u32,
    pub time: f64,
    pub etype: EType,
    pub cell: Cell,
    pub ch: Option<usize>, // Specifies ch in use for END events
    // Specifies hand-off arrival cell for END-events that are immediately followed by HOFF
    pub to_cell: Option<Cell>,
}

/// Event Identifiers
struct EI {
    // floats must be wrapped in NotNaN to support ordering (impl Ord)
    // NotNaNs must be wrapped in RevOrd which reveres ordering since BinaryHeap sorts by max
    time: RevOrd<NotNaN<f64>>,
    id: RevOrd<u32>,
}

impl Ord for EI {
    fn cmp(&self, other: &EI) -> Ordering {
        let o = self.time.cmp(&other.time);
        match o {
            Ordering::Equal => self.id.cmp(&other.id),
            _ => o,
        }
    }
}

impl PartialOrd for EI {
    fn partial_cmp(&self, other: &EI) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for EI {
    fn eq(&self, other: &EI) -> bool {
        self.id == other.id
    }
}
impl Eq for EI {}

#[derive(Default)]
pub struct EventGen {
    id: u32,                                      // Current Event ID
    call_rate: f32,                               // Call rate, calls per minutes
    call_dur_inv: f32,                            // (Inverse of) Average call duration, minutes
    hoff_call_dur_inv: f32, // (Inverse of) Average hand-off call duration, minutes
    event_pq: BinaryHeap<EI>, // Min-heap of event-identifiers sorted on event times
    events: HashMap<u32, Event>, // Mapping from event IDs to event structs
    end_ids: HashMap<(usize, usize, usize), u32>, // Mapping from cell-channel pairs to end event IDs
}

impl EventGen {
    pub fn new(opt: &Opt) -> EventGen {
        let call_rate = opt.call_rate_ph / 60.0;
        debug!(
            "Call intertime: {}, call duration: {}",
            call_rate, opt.call_dur
        );
        EventGen {
            call_rate: call_rate,
            call_dur_inv: 1.0 / opt.call_dur,
            hoff_call_dur_inv: 1.0 / opt.hoff_call_dur,
            ..Default::default()
        }
    }

    pub fn push(&mut self, event: Event) {
        debug!("Pushed event: {:?}", event);
        if event.etype == EType::END {
            let c = event.cell.clone();
            self.end_ids.insert(
                (c.row, c.col, event.ch.expect("No CH for end event").clone()),
                event.id.clone(),
            );
        }
        unsafe {
            // 'event.time' was just generated by one of the 'event_*' functions
            // and cannot be NaN
            let t: NotNaN<f64> = NotNaN::unchecked_new(event.time.clone());
            self.event_pq.push(EI {
                time: RevOrd { 0: t },
                id: RevOrd {
                    0: event.id.clone(),
                },
            });
        }
        self.events.insert(event.id, event);
    }

    pub fn pop(&mut self) -> Event {
        let ei = self.event_pq.pop().expect("No events to pop");
        let event = self.events
            .remove(&ei.id.0)
            .expect("Event for ID not found");
        if event.etype == EType::END {
            self.end_ids
                .remove(&(
                    event.cell.row,
                    event.cell.col,
                    event.ch.expect("No CH for end event"),
                ))
                .expect("End ID not found");
        }
        event
    }

    pub fn reassign(&mut self, cell: Cell, from_ch: usize, to_ch: usize) {
        assert_ne!(from_ch, to_ch);
        let id = self.end_ids
            .remove(&(cell.row, cell.col, from_ch))
            .expect("End ID not found");
        self.end_ids.insert((cell.row, cell.col, to_ch), id);
        self.events.get_mut(&id).expect("Event for ID not found").ch = Some(to_ch);
    }

    pub fn event_new(&mut self, t: f64, cell: Cell) {
        let dt = Exp::new(self.call_rate.into()).sample(&mut thread_rng()) as f64;
        self.id += 1;
        let event = Event {
            id: self.id,
            time: t + dt,
            etype: EType::NEW,
            cell: cell,
            ch: None,
            to_cell: None,
        };
        self.push(event)
    }

    /// Hand off a call to a neighboring cell 'neigh' picked randomly at uniform from 'neighs'.
    /// The hand-off from 'cell' is deconstructed into two parts: the departure from 'cell',
    /// and the subsequent arrival in 'neigh'. These two events have the same time stamp, though
    /// since the ID of the arrival is larger it will be handled last.
    pub fn event_hoff_new(&mut self, t: f64, cell: Cell, ch: usize) {
        let neighs = neighbors(1, cell.row, cell.col, false);
        let neigh_i: usize = Uniform::from(0..neighs.rows()).sample(&mut thread_rng());
        let to_cell = cell_of(neighs, neigh_i);
        let dur_inv = self.call_dur_inv.into();
        let end_t = self._event_end(t, dur_inv, cell, ch, Some(to_cell.clone()));
        self.id += 1;
        let new_event = Event {
            id: self.id,
            time: end_t,
            etype: EType::HOFF,
            cell: to_cell,
            ch: None,
            to_cell: None,
        };
        self.push(new_event)
    }

    /// Generate the departure event of a regular call
    pub fn event_end(&mut self, t: f64, cell: Cell, ch: usize) -> f64 {
        let dur = self.call_dur_inv.into();
        self._event_end(t, dur, cell, ch, None)
    }

    /// Generate the departure event of a handed-off call
    pub fn event_hoff_end(&mut self, t: f64, cell: Cell, ch: usize) -> f64 {
        let dur_inv = self.hoff_call_dur_inv.into();
        self._event_end(t, dur_inv, cell, ch, None)
    }

    fn _event_end(
        &mut self,
        t: f64,
        dur_inv: f64,
        cell: Cell,
        ch: usize,
        to_cell: Option<Cell>,
    ) -> f64 {
        let dt = Exp::new(dur_inv).sample(&mut thread_rng()) as f64;
        self.id += 1;
        let event = Event {
            id: self.id,
            time: t + dt,
            etype: EType::END,
            cell: cell,
            ch: Some(ch),
            to_cell: to_cell,
        };
        self.push(event);
        t + dt
    }
}
