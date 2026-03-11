"""Tests for the Medical Diagnostics Agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from medical_diagnostics_agent.main import handler


@pytest.mark.asyncio
async def test_handler_returns_response():
    """Test that handler accepts messages and returns a response."""
    messages = [{"role": "user", "content": "Patient presents with chest pain and shortness of breath..."}]

    # Mock the run_agent function to return a mock response
    mock_response = MagicMock()
    mock_response.run_id = "test-run-id"
    mock_response.status = "COMPLETED"

    # Mock _initialized to skip initialization and run_agent to return our mock
    with (
        patch("medical_diagnostics_agent.main._initialized", True),
        patch("medical_diagnostics_agent.main.run_agent", new_callable=AsyncMock, return_value=mock_response),
    ):
        result = await handler(messages)

    # Verify we get a result back
    assert result is not None
    assert result.run_id == "test-run-id"
    assert result.status == "COMPLETED"


@pytest.mark.asyncio
async def test_handler_with_multiple_messages():
    """Test that handler processes multiple messages correctly."""
    messages = [
        {"role": "system", "content": "You are a medical diagnostics assistant."},
        {"role": "user", "content": "Analyze this medical report: Patient has elevated heart rate..."},
    ]

    mock_response = MagicMock()
    mock_response.run_id = "test-run-id-2"

    with (
        patch("medical_diagnostics_agent.main._initialized", True),
        patch(
            "medical_diagnostics_agent.main.run_agent", new_callable=AsyncMock, return_value=mock_response
        ) as mock_run,
    ):
        result = await handler(messages)

    # Verify run_agent was called
    mock_run.assert_called_once_with(messages)
    assert result is not None
    assert result.run_id == "test-run-id-2"


@pytest.mark.asyncio
async def test_handler_initialization():
    """Test that handler initializes on first call."""
    messages = [{"role": "user", "content": "Patient with diabetes symptoms..."}]

    mock_response = MagicMock()

    # Start with _initialized as False to test initialization path
    with (
        patch("medical_diagnostics_agent.main._initialized", False),
        patch("medical_diagnostics_agent.main.initialize_agent", new_callable=AsyncMock) as mock_init,
        patch(
            "medical_diagnostics_agent.main.run_agent", new_callable=AsyncMock, return_value=mock_response
        ) as mock_run,
        patch("medical_diagnostics_agent.main._init_lock", new_callable=MagicMock()) as mock_lock,
    ):
        # Configure the lock to work as an async context manager
        mock_lock_instance = MagicMock()
        mock_lock_instance.__aenter__ = AsyncMock(return_value=None)
        mock_lock_instance.__aexit__ = AsyncMock(return_value=None)
        mock_lock.return_value = mock_lock_instance

        result = await handler(messages)

        # Verify initialization was called
        mock_init.assert_called_once()
        # Verify run_agent was called
        mock_run.assert_called_once_with(messages)
        # Verify we got a result
        assert result is not None


@pytest.mark.asyncio
async def test_handler_race_condition_prevention():
    """Test that handler prevents race conditions with initialization lock."""
    messages = [{"role": "user", "content": "Emergency medical case..."}]

    mock_response = MagicMock()

    # Test with multiple concurrent calls
    with (
        patch("medical_diagnostics_agent.main._initialized", False),
        patch("medical_diagnostics_agent.main.initialize_agent", new_callable=AsyncMock) as mock_init,
        patch("medical_diagnostics_agent.main.run_agent", new_callable=AsyncMock, return_value=mock_response),
        patch("medical_diagnostics_agent.main._init_lock", new_callable=MagicMock()) as mock_lock,
    ):
        # Configure the lock to work as an async context manager
        mock_lock_instance = MagicMock()
        mock_lock_instance.__aenter__ = AsyncMock(return_value=None)
        mock_lock_instance.__aexit__ = AsyncMock(return_value=None)
        mock_lock.return_value = mock_lock_instance

        # Call handler twice to ensure lock is used
        await handler(messages)
        await handler(messages)

        # Verify initialize_agent was called only once (due to lock)
        mock_init.assert_called_once()


@pytest.mark.asyncio
async def test_handler_with_medical_query():
    """Test that handler can process a medical diagnostics query."""
    messages = [
        {
            "role": "user",
            "content": "Patient presents with chest pain, heart palpitations, shortness of breath, dizziness, and sweating episodes over the past 3 months. ECG shows normal sinus rhythm, cardiac enzymes normal. Please provide comprehensive medical diagnosis analysis.",
        }
    ]

    mock_response = MagicMock()
    mock_response.run_id = "medical-diagnosis-run-id"
    mock_response.content = "### Final Diagnosis:\n\n- **Panic Disorder**: The symptoms are consistent with panic attacks...\n- **GERD**: Contributing factor...\n- **Anxiety-Induced Hyperventilation**: Secondary respiratory component..."

    with (
        patch("medical_diagnostics_agent.main._initialized", True),
        patch("medical_diagnostics_agent.main.run_agent", new_callable=AsyncMock, return_value=mock_response),
    ):
        result = await handler(messages)

    assert result is not None
    assert result.run_id == "medical-diagnosis-run-id"
    assert "Panic Disorder" in result.content
    assert "GERD" in result.content
    assert "Anxiety-Induced Hyperventilation" in result.content


@pytest.mark.asyncio
async def test_handler_with_complex_medical_case():
    """Test that handler can process complex multi-system medical cases."""
    messages = [
        {
            "role": "user",
            "content": "29-year-old male with anxiety disorder, GERD, experiencing chest pain and palpitations. Recent ECG normal, cardiac enzymes normal, Holter monitor shows occasional PVCs. High-stress job as investment banker. Current medications: Lorazepam 0.5mg as needed, Omeprazole 20mg daily. Provide multi-specialist analysis.",
        }
    ]

    mock_response = MagicMock()
    mock_response.run_id = "complex-case-run-id"
    mock_response.content = "Multi-specialist analysis completed: Cardiologist reports normal cardiac function, Psychologist identifies panic disorder, Pulmonologist notes anxiety-induced respiratory symptoms."

    with (
        patch("medical_diagnostics_agent.main._initialized", True),
        patch("medical_diagnostics_agent.main.run_agent", new_callable=AsyncMock, return_value=mock_response),
    ):
        result = await handler(messages)

    assert result is not None
    assert result.run_id == "complex-case-run-id"
    assert "Cardiologist" in result.content
    assert "Psychologist" in result.content
    assert "Pulmonologist" in result.content


@pytest.mark.asyncio
async def test_handler_error_handling():
    """Test that handler handles errors gracefully."""
    messages = [{"role": "user", "content": "Invalid medical data..."}]

    with (
        patch("medical_diagnostics_agent.main._initialized", True),
        patch(
            "medical_diagnostics_agent.main.run_agent",
            new_callable=AsyncMock,
            side_effect=Exception("Medical analysis failed"),
        ),
        pytest.raises(Exception, match="Medical analysis failed"),
    ):
        await handler(messages)


@pytest.mark.asyncio
async def test_handler_with_empty_message():
    """Test that handler handles empty messages."""
    messages = [{"role": "user", "content": ""}]

    mock_response = MagicMock()
    mock_response.run_id = "empty-message-run-id"
    mock_response.content = "Please provide a valid medical report for analysis."

    with (
        patch("medical_diagnostics_agent.main._initialized", True),
        patch("medical_diagnostics_agent.main.run_agent", new_callable=AsyncMock, return_value=mock_response),
    ):
        result = await handler(messages)

    assert result is not None
    assert result.run_id == "empty-message-run-id"
    assert "valid medical report" in result.content
