import pytest
import torch
from unittest.mock import Mock, MagicMock
from btsft.trainers.blurred_thoughts import BlurredThoughtsSFTTrainer


class TestMaskedCritiqueTrainer:
    """Test cases for the Masked Critique Trainer."""
    
    def test_trainer_initialization(self):
        """Test that the trainer initializes correctly."""
        trainer = BlurredThoughtsSFTTrainer(
            model=Mock(),
            args=Mock(),
            tokenizer=Mock(),
            bf_beta=0.05,
            format_reward_func=Mock()
        )
        
        assert trainer.bf_beta == 0.05
        assert trainer.format_reward_func is not None
    
    def test_compute_loss_with_rewards(self):
        """Test loss computation with format rewards."""
        # Mock model outputs
        mock_outputs = Mock()
        mock_outputs.loss = torch.tensor(1.0)
        
        # Mock model
        mock_model = Mock()
        mock_model.return_value = mock_outputs
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.batch_decode.return_value = ["<think><critique>test</critique></think>"]
        
        # Mock format reward function
        mock_reward_func = Mock()
        mock_reward_func.return_value = [1.0]  # Perfect format adherence
        
        trainer = BlurredThoughtsSFTTrainer(
            model=mock_model,
            args=Mock(),
            tokenizer=mock_tokenizer,
            bf_beta=0.05,
            format_reward_func=mock_reward_func
        )
        
        # Mock inputs
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "labels": torch.tensor([[1, 2, 3]])
        }
        
        loss = trainer.compute_loss(mock_model, inputs)
        
        # Should be close to the original loss since reward is perfect
        assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)
    
    def test_compute_loss_with_imperfect_rewards(self):
        """Test loss computation with imperfect format rewards."""
        # Mock model outputs
        mock_outputs = Mock()
        mock_outputs.loss = torch.tensor(1.0)
        
        # Mock model
        mock_model = Mock()
        mock_model.return_value = mock_outputs
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.batch_decode.return_value = ["<think><critique>test</critique></think>"]
        
        # Mock format reward function - imperfect format
        mock_reward_func = Mock()
        mock_reward_func.return_value = [0.0]  # No format adherence
        
        trainer = BlurredThoughtsSFTTrainer(
            model=mock_model,
            args=Mock(),
            tokenizer=mock_tokenizer,
            bf_beta=0.05,
            format_reward_func=mock_reward_func
        )
        
        # Mock inputs
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "labels": torch.tensor([[1, 2, 3]])
        }
        
        loss = trainer.compute_loss(mock_model, inputs)
        
        # Should be higher than original loss due to format penalty
        assert loss > torch.tensor(1.0)


if __name__ == "__main__":
    pytest.main([__file__])
