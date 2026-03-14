from unittest import TestCase
import torch

from src.ddpm import BetaSchedule


class TestBetaSchedule(TestCase):

    ABSOLUTE_TOLERANCE = 1e-10

    def setUp(self):
        self.bs = BetaSchedule(betas=torch.Tensor([0.1, 0.2, 0.3]))

    def tearDown(self):
        del self.bs

    def test_len(self):
        self.assertEqual(
            first=len(self.bs),
            second=3
        )

    def test_betas(self):
        idx1 = torch.Tensor([2, 1, 0]).to(torch.int32)
        betas = self.bs.beta(t=idx1)
        self.assertTrue(
            torch.all(
                betas == torch.Tensor([0.3, 0.2, 0.1]))
        )

        idx2 = torch.Tensor([[1, 0], [2, 1]]).to(torch.int32)
        betas = self.bs.beta(t=idx2)
        self.assertTrue(
            torch.all(
                betas == torch.Tensor([[0.2, 0.1], [0.3, 0.2]]))
        )

        idx3 = 1
        betas = self.bs.beta(t=idx3)
        self.assertTrue(
            torch.all(
                betas == torch.Tensor([0.2])
            )
        )
        self.assertIsInstance(betas, torch.Tensor)

    def test_alphas(self):
        # alphas: [0.9, 0.8, 0.7]
        idx1 = torch.Tensor([2, 1, 0]).to(torch.int32)
        alphas = self.bs.alpha(t=idx1)
        self.assertTrue(
            torch.all(
                alphas == torch.Tensor([0.7, 0.8, 0.9]))
        )

        idx2 = torch.Tensor([[1, 0], [2, 1]]).to(torch.int32)
        alphas = self.bs.alpha(t=idx2)
        self.assertTrue(
            torch.all(
                alphas == torch.Tensor([[0.8, 0.9], [0.7, 0.8]]))
        )

        idx3 = 1
        alphas = self.bs.alpha(t=idx3)
        self.assertTrue(
            torch.all(
                alphas == torch.Tensor([0.8])
            )
        )
        self.assertIsInstance(alphas, torch.Tensor)

    def test_alphas_bar(self):
        # alphas bar: [0.9, 0.72, 0.504]
        idx1 = torch.Tensor([2, 1, 0]).to(torch.int32)
        alphas_bar = self.bs.alpha_bar(t=idx1)
        self.assertTrue(
            torch.allclose(
                alphas_bar,
                torch.Tensor([0.504, 0.72, 0.9]),
                atol=self.ABSOLUTE_TOLERANCE
            )
        )

        idx2 = torch.Tensor([[1, 0], [2, 1]]).to(torch.int32)
        alphas_bar = self.bs.alpha_bar(t=idx2)
        self.assertTrue(
            torch.allclose(
                alphas_bar,
                torch.Tensor([[0.72, 0.9], [0.504, 0.72]]),
                atol=self.ABSOLUTE_TOLERANCE
            )
        )

        idx3 = 1
        alphas_bar = self.bs.alpha_bar(t=idx3)
        self.assertTrue(
            torch.allclose(
                alphas_bar,
                torch.Tensor([0.72]),
                atol=self.ABSOLUTE_TOLERANCE
            )
        )
        self.assertIsInstance(alphas_bar, torch.Tensor)

    def test_sqrt_alphas(self):
        # sqrt alphas [0.94868329805, 0.894427191, 0.83666002653]
        idx1 = torch.Tensor([2, 1, 0]).to(torch.int32)
        sqrt_alphas = self.bs.sqrt_alpha(t=idx1)
        self.assertTrue(
            torch.allclose(
                sqrt_alphas,
                torch.Tensor([0.83666002653, 0.894427191, 0.94868329805]),
                atol=self.ABSOLUTE_TOLERANCE
            )
        )

        idx2 = torch.Tensor([[1, 0], [2, 1]]).to(torch.int32)
        sqrt_alphas = self.bs.sqrt_alpha(t=idx2)
        self.assertTrue(
            torch.allclose(
                sqrt_alphas,
                torch.Tensor([[0.894427191, 0.94868329805], [0.83666002653, 0.894427191]]),
                atol=self.ABSOLUTE_TOLERANCE
            )
        )

        idx3 = 1
        sqrt_alphas = self.bs.sqrt_alpha(t=idx3)
        self.assertTrue(
            torch.allclose(
                sqrt_alphas,
                torch.Tensor([0.894427191]),
                atol=self.ABSOLUTE_TOLERANCE
            )
        )
        self.assertIsInstance(sqrt_alphas, torch.Tensor)

    def test_sqrt_alphas_bar(self):
        # sqrt alphas bar: [0.94868329805, 0.84852813742, 0.70992957397]
        idx1 = torch.Tensor([2, 1, 0]).to(torch.int32)
        sqrt_alpha_bar = self.bs.sqrt_alpha_bar(t=idx1)
        self.assertTrue(
            torch.allclose(
                sqrt_alpha_bar,
                torch.Tensor([0.70992957397, 0.84852813742, 0.94868329805]),
                atol=self.ABSOLUTE_TOLERANCE
            )
        )

        idx2 = torch.Tensor([[1, 0], [2, 1]]).to(torch.int32)
        sqrt_alpha_bar = self.bs.sqrt_alpha_bar(t=idx2)
        self.assertTrue(
            torch.allclose(
                sqrt_alpha_bar,
                torch.Tensor([[0.84852813742, 0.94868329805], [0.70992957397, 0.84852813742]]),
                atol=self.ABSOLUTE_TOLERANCE
            )
        )

        idx3 = 1
        sqrt_alpha_bar = self.bs.sqrt_alpha_bar(t=idx3)
        self.assertTrue(
            torch.allclose(
                sqrt_alpha_bar,
                torch.Tensor([0.84852813742]),
                atol=self.ABSOLUTE_TOLERANCE
            )
        )
        self.assertIsInstance(sqrt_alpha_bar, torch.Tensor)

    def test_sqrt_one_minus_alphas_bar(self):
        # sqrt one minus alphas bar: [0.31622776601, 0.52915026221, 0.70427267446]
        idx1 = torch.Tensor([2, 1, 0]).to(torch.int32)
        sqrt_one_minus_alphas_bar = self.bs.sqrt_one_minus_alpha_bar(t=idx1)
        self.assertTrue(
            torch.allclose(
                sqrt_one_minus_alphas_bar,
                torch.Tensor([0.70427267446, 0.52915026221, 0.31622776601]),
                atol=self.ABSOLUTE_TOLERANCE
            )
        )

        idx2 = torch.Tensor([[1, 0], [2, 1]]).to(torch.int32)
        sqrt_one_minus_alphas_bar = self.bs.sqrt_one_minus_alpha_bar(t=idx2)
        self.assertTrue(
            torch.allclose(
                sqrt_one_minus_alphas_bar,
                torch.Tensor([[0.52915026221, 0.31622776601], [0.70427267446, 0.52915026221]]),
                atol=self.ABSOLUTE_TOLERANCE
            )
        )

        idx3 = 1
        sqrt_one_minus_alphas_bar = self.bs.sqrt_one_minus_alpha_bar(t=idx3)
        self.assertTrue(
            torch.allclose(
                sqrt_one_minus_alphas_bar,
                torch.Tensor([0.52915026221]),
                atol=self.ABSOLUTE_TOLERANCE
            )
        )
        self.assertIsInstance(sqrt_one_minus_alphas_bar, torch.Tensor)

    def test_sigma(self):
        # sigmas: [0, 0.26726124191, 0.41152744587]
        idx1 = torch.Tensor([2, 1, 0]).to(torch.int32)
        sigmas = self.bs.sigma(t=idx1)
        self.assertTrue(
            torch.allclose(
                sigmas,
                torch.Tensor([0.41152744587, 0.26726124191, 0.0]),
                atol=self.ABSOLUTE_TOLERANCE
            )
        )

        idx2 = torch.Tensor([[1, 0], [2, 1]]).to(torch.int32)
        sigmas = self.bs.sigma(t=idx2)
        self.assertTrue(
            torch.allclose(
                sigmas,
                torch.Tensor([[0.26726124191, 0.0], [0.41152744587, 0.26726124191]]),
                atol=self.ABSOLUTE_TOLERANCE
            )
        )

        idx3 = 1
        sigmas = self.bs.sigma(t=idx3)
        self.assertTrue(
            torch.allclose(
                sigmas,
                torch.Tensor([0.26726124191]),
                atol=self.ABSOLUTE_TOLERANCE
            )
        )
        self.assertIsInstance(sigmas, torch.Tensor)
