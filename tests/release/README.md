# Release and Repository-Health Tests

This folder contains tests that protect package metadata, README/docs hygiene,
public helper coverage, and release-facing examples.

These tests should stay fast and deterministic. They may inspect repository
files and examples, but they should not launch long VMEC solves.
