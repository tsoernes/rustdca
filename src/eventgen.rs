use gridfuncs::*;
use ndarray::prelude::*;
use ordered_float::*;
use rand::distributions::{Distribution, Exp, Uniform};
use rand::thread_rng;
use revord::RevOrd;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;

pub struct Event {
    pub id: u32,
    pub time: f32,
    pub etype: EType,
    pub cell: Cell,
    pub ch: Option<usize>,
}

// Event identifiers
struct EI {
    // floats must be wrapped in NotNaN to support ordering (impl Ord)
    // NotNaNs must be wrapped in RevOrd which reveres ordering since BinaryHeap sorts by max
    time: RevOrd<NotNaN<f32>>,
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

pub struct EventGen {
    // Current Event ID
    id: u32,
    call_intertime: f32,
    call_dur: f32,
    hoff_call_dur: f32,
    // Min-heap of event-identifiers sorted on event times
    event_pq: BinaryHeap<EI>,
    // Mapping from event IDs to event structs
    events: HashMap<u32, Event>,
    // Mapping from cell-channel pairs to end event IDs
    end_ids: HashMap<(usize, usize, usize), u32>,
}

impl EventGen {
    pub fn push(&mut self, event: Event) {
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
            let t: NotNaN<f32> = NotNaN::unchecked_new(event.time.clone());
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

    pub fn peek(&mut self) -> &Event {
        let ei = self.event_pq.peek().expect("No events to peek");
        self.events.get(&ei.id.0).expect("Event for ID not found")
    }

    pub fn reassign(&mut self, cell: Cell, from_ch: usize, to_ch: usize) {
        assert_ne!(from_ch, to_ch);
        let id = self.end_ids
            .remove(&(cell.row, cell.col, from_ch))
            .expect("End ID not found");
        // Is the right struct field changed?
        self.events.get_mut(&id).expect("Event for ID not found").ch = Some(to_ch);
        self.end_ids.insert((cell.row, cell.col, to_ch), id);
    }

    pub fn event_new(&mut self, t: f32, cell: Cell) {
        let dt: f32 = Exp::new(self.call_intertime.into()).sample(&mut thread_rng()) as f32;
        self.id += 1;
        let event = Event {
            id: self.id,
            time: t + dt,
            etype: EType::NEW,
            cell: cell,
            ch: None,
        };
        self.push(event)
    }

    /// Hand off a call to a neighboring cell 'neigh' picked randomly at uniform from 'neighs'.
    /// The hand-off from 'cell' is deconstructed into two parts: the departure from 'cell',
    /// and the subsequent arrival in 'neigh'. These two events have the same time stamp, though
    /// since the ID of the arrival is larger it will be handled last.
    pub fn event_hoff_new(
        &mut self,
        t: f32,
        cell: Cell,
        ch: usize,
        neighs: ArrayView<'static, usize, Ix2>,
    ) {
        let end_t = self.event_end(t, cell, ch);
        let neigh_i: usize = Uniform::from(0..neighs.len()).sample(&mut thread_rng());
        self.id += 1;
        let new_event = Event {
            id: self.id,
            time: end_t,
            etype: EType::HOFF,
            cell: cell_of(neighs, neigh_i),
            ch: None,
        };
        self.push(new_event)
    }

    fn _event_end(&mut self, t: f32, dt: f32, cell: Cell, ch: usize) -> f32 {
        self.id += 1;
        let event = Event {
            id: self.id,
            time: t + dt,
            etype: EType::END,
            cell: cell,
            ch: Some(ch),
        };
        self.push(event);
        t + dt
    }

    pub fn event_end(&mut self, t: f32, cell: Cell, ch: usize) -> f32 {
        let dt: f32 = Exp::new(self.call_dur.into()).sample(&mut thread_rng()) as f32;
        self._event_end(t, dt, cell, ch)
    }

    pub fn event_hoff_end(&mut self, t: f32, cell: Cell, ch: usize) -> f32 {
        let dt: f32 = Exp::new(self.hoff_call_dur.into()).sample(&mut thread_rng()) as f32;
        self._event_end(t, dt, cell, ch)
    }
}
