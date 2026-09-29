// VERIFYX AI - Main Frontend Utilities & Micro-Interactions

document.addEventListener('DOMContentLoaded', () => {
    // Tooltips activation
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    console.log("VERIFYX AI System Initialized. Prototype ready.");
});
