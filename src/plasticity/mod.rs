pub mod general;
pub mod drucker_prager_classic;

// Re-export everything from both modules for backward compatibility
pub use general::*;
pub use drucker_prager_classic::*;
