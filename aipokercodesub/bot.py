import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from player import Player, PlayerAction
from hand_evaluator import HandEvaluator, HandRank
from typing import Tuple


SUIT_REMAP = {0: 3, 1: 2, 2: 1, 3: 0}

def decode_card(card_index: int):

    
    if card_index == 0:
        return 13, 4  # unknown card
    # card_index was generated as (suit.value * 13) + rank.value - 1.
    suit_val = card_index // 13   # from new system: 0=SPADES, 1=HEARTS, 2=DIAMONDS, 3=CLUBS
    rank_val = (card_index % 13) + 2  # because rank TWO is 2, ... ACE is 14
    rank = rank_val - 2  # now 0-12
    suit = SUIT_REMAP.get(suit_val, 4)
    return rank, suit

def hand_rank_to_numeric(hand_rank: HandRank):
    """
    Map HandRank (from HandEvaluator) to a numeric value matching your original COMBINATION_MAP.
    Original mapping: High Card:1, One Pair:2, Two Pair:3, Three of a Kind:4,
                      Straight:5, Flush:6, Full House:7, Four of a Kind:8, Straight Flush:9,
                      Royal Flush is mapped to 9 as well.
    """
    mapping = {
        HandRank.HIGH_CARD: 1,
        HandRank.PAIR: 2,
        HandRank.TWO_PAIR: 3,
        HandRank.THREE_OF_A_KIND: 4,
        HandRank.STRAIGHT: 5,
        HandRank.FLUSH: 6,
        HandRank.FULL_HOUSE: 7,
        HandRank.FOUR_OF_A_KIND: 8,
        HandRank.STRAIGHT_FLUSH: 9,
        HandRank.ROYAL_FLUSH: 9
    }
    return mapping.get(hand_rank, 0)

# Neural network model (unchanged from your original code)
class CombinedModel(nn.Module):
    def __init__(self, input_size, hidden_size_cls, output_size_cls, regression_output_size):
        super(CombinedModel, self).__init__()
        # Classification branch
        self.fc1_cls = nn.Linear(input_size, hidden_size_cls)
        self.bn1_cls = nn.BatchNorm1d(hidden_size_cls)
        self.dropout1_cls = nn.Dropout(0.3)
        self.fc2_cls = nn.Linear(hidden_size_cls, hidden_size_cls)
        self.bn2_cls = nn.BatchNorm1d(hidden_size_cls)
        self.dropout2_cls = nn.Dropout(0.3)
        self.fc3_cls = nn.Linear(hidden_size_cls, hidden_size_cls)
        self.bn3_cls = nn.BatchNorm1d(hidden_size_cls)
        self.dropout3_cls = nn.Dropout(0.3)
        self.fc4_cls = nn.Linear(hidden_size_cls, output_size_cls)
        
        # Regression branch
        self.fc1_reg = nn.Linear(input_size, 32)
        self.fc2_reg = nn.Linear(32, 64)
        self.fc3_reg = nn.Linear(64, 128)
        self.fc35_reg = nn.Linear(128, 32)
        self.fc375_reg = nn.Linear(32, 16)
        self.fc4_reg = nn.Linear(16, regression_output_size)

    def forward(self, x):
        # Classification branch
        x_cls = torch.relu(self.bn1_cls(self.fc1_cls(x)))
        x_cls = self.dropout1_cls(x_cls)
        x_cls = torch.relu(self.bn2_cls(self.fc2_cls(x_cls)))
        x_cls = self.dropout2_cls(x_cls)
        x_cls = torch.relu(self.bn3_cls(self.fc3_cls(x_cls)))
        x_cls = self.dropout3_cls(x_cls)
        x_cls = self.fc4_cls(x_cls)
        
        # Regression branch
        x_reg = torch.relu(self.fc1_reg(x))
        x_reg = torch.relu(self.fc2_reg(x_reg))
        x_reg = torch.relu(self.fc3_reg(x_reg))
        x_reg = torch.relu(self.fc35_reg(x_reg))
        x_reg = torch.relu(self.fc375_reg(x_reg))
        x_reg = self.fc4_reg(x_reg)
        
        return x_cls, x_reg

# Our optimized poker bot for the new system
class OptimizedPokerPlayer(Player):
    def __init__(self, name: str, stack: int, seat_index: int = 0):
        super().__init__(name, stack)
        self.seat_index = seat_index  # your seat position in the table (if known)
        # Initialize network parameters as before
        input_size = 26
        hidden_size_cls = 64
        output_size_cls = 4  # mapping: 0: fold, 1: call, 2: raise, 3: bet/call alternative
        regression_output_size = 1
        self.model = CombinedModel(input_size, hidden_size_cls, output_size_cls, regression_output_size)
        # Load pretrained model weights (ensure model.pth is available in the working directory)
        self.model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
        self.model.eval()
    
    def action(self, game_state: list, action_history: list) -> Tuple[PlayerAction, int]:
        """
        The game_state list is structured as follows (see game.py get_game_state):
          [hole_card1, hole_card2, community_card1, community_card2, community_card3,
           community_card4, community_card5, pot, current_bet, big_blind,
           active_player_index, num_players, stack1, stack2, ..., hand_number]
        We'll extract our hole cards and community cards (decoding them),
        then build a 26-dim input vector as expected by our model.
        """
        # --- Extract basic game info ---
        num_players = game_state[11]
        pot = game_state[7]
        current_bet = game_state[8]
        big_blind = game_state[9]
        
        # Our player's current bet is stored in self.bet_amount.
        call_amount = current_bet - self.bet_amount
        if call_amount < 0:
            call_amount = 0
        
        # --- Decode our hole cards ---
        # game_state[0] and [1] are our hole card indices.
        hole_card1_idx = game_state[0]
        hole_card2_idx = game_state[1]
        card1_rank, card1_suit = decode_card(hole_card1_idx)
        card2_rank, card2_suit = decode_card(hole_card2_idx)
        # For our combination feature we use HandEvaluator.
        # (In a real game, self.hole_cards would be set when cards are dealt.
        # Here we reconstruct from game_state.)
        from card import Card, Suit, Rank
        def idx_to_card(idx):
            if idx == 0:
                return None
            # Reverse decoding:
            suit_val = idx // 13  # new system suit value
            rank_val = (idx % 13) + 2
            # Map suit to the new system enum:
            suit_enum = {0: Suit.SPADES, 1: Suit.HEARTS, 2: Suit.DIAMONDS, 3: Suit.CLUBS}.get(suit_val)
            rank_enum = {v: k for k, v in {Rank.TWO:2, Rank.THREE:3, Rank.FOUR:4, Rank.FIVE:5,
                                           Rank.SIX:6, Rank.SEVEN:7, Rank.EIGHT:8, Rank.NINE:9,
                                           Rank.TEN:10, Rank.JACK:11, Rank.QUEEN:12, Rank.KING:13, Rank.ACE:14}.items()}.get(rank_val)
            # If for some reason rank_enum is None, default to TWO.
            if rank_enum is None:
                rank_enum = Rank.TWO
            return Card(rank_enum, suit_enum)
        
        hole_cards = []
        for idx in [hole_card1_idx, hole_card2_idx]:
            card = idx_to_card(idx)
            if card:
                hole_cards.append(card)
        
        # --- Decode community cards (indices 2 to 6) ---
        community_cards = []
        for i in range(2, 7):
            idx = game_state[i]
            if idx != 0:
                card = idx_to_card(idx)
                if card:
                    community_cards.append(card)
        
        # Evaluate hand combination using the new system's HandEvaluator.
        hand_result = HandEvaluator.evaluate_hand(hole_cards, community_cards)
        combination_numeric = hand_rank_to_numeric(hand_result.hand_rank)
        
        # Determine round from number of community cards:
        if len(community_cards) == 0:
            round_encoded = 0  # pre-flop
        elif len(community_cards) == 3:
            round_encoded = 1  # flop
        else:
            round_encoded = 2  # turn/river
        
        # For opponents' actions, bets, stacks – new system does not provide these per opponent.
        # We'll fill with default values (using 4 to denote 'x'/unknown and 0 for amounts).
        default_action = 4  # unknown
        default_stack = 0
        default_bet = 0
        
        # Build the 26-dimensional input vector.
        # Layout (as per your original network):
        # [ playing, seat_position, stack, round, pot,
        #   p1_action, p1_stack, p1_bet, p2_action, p2_stack, p2_bet,
        #   combination,
        #   hole_card1: rank, hole_card1: suit, hole_card2: rank, hole_card2: suit,
        #   community card1: rank, community card1: suit, ... up to community card5 ]
        input_vector = []
        input_vector.append(num_players)            # playing
        input_vector.append(self.seat_index)          # seat_position
        input_vector.append(self.stack)               # our stack
        input_vector.append(round_encoded)            # round
        input_vector.append(pot)                      # pot
        input_vector.append(default_action)           # p1_action
        input_vector.append(default_stack)            # p1_stack
        input_vector.append(default_bet)              # p1_bet
        input_vector.append(default_action)           # p2_action
        input_vector.append(default_stack)            # p2_stack
        input_vector.append(default_bet)              # p2_bet
        input_vector.append(combination_numeric)      # combination
        
        # Hole cards (order: card1 then card2)
        input_vector.append(card1_rank)   # hole card1 rank
        input_vector.append(card1_suit)   # hole card1 suit
        input_vector.append(card2_rank)   # hole card2 rank
        input_vector.append(card2_suit)   # hole card2 suit
        
        # Community cards: expect 5 cards, each as (rank, suit)
        # game_state indices 2 to 6 correspond to these cards.
        for i in range(2, 7):
            idx = game_state[i]
            rank_val, suit_val = decode_card(idx)
            input_vector.append(rank_val)
            input_vector.append(suit_val)
        
        # Convert input vector to tensor.
        input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
        # Run the model.
        with torch.no_grad():
            predictions_cls, predictions_reg = self.model(input_tensor)
        softmax_preds = F.softmax(predictions_cls, dim=1)
        action_idx = int(np.argmax(softmax_preds.detach().numpy()))
        
        # Decision mapping:
        # 0: Fold, 1: Call (or Check if no call needed), 2: Raise, 3: (Alternate Call/BET) – here we map to call.
        if action_idx == 0:
            return PlayerAction.FOLD, 0
        elif action_idx in [1, 3]:
            # If there is an amount to call, then call; otherwise check.
            if call_amount > 0:
                return PlayerAction.CALL, call_amount
            else:
                return PlayerAction.CHECK, 0
        elif action_idx == 2:
            # Raise: use the regression output to determine raise amount.
            reg_val = predictions_reg.detach().numpy()[0][0]
            # Scale factor: similar logic to original (you may adjust these factors).
            factor = 100 if int(reg_val/100) == 0 else 1000
            # Determine minimum raise: must at least call plus big blind.
            min_raise = call_amount + big_blind
            # Maximum possible is our current stack.
            maximum = self.stack
            # Scale the regression output to the available range.
            additional = int((reg_val / factor) * (maximum - min_raise))
            raise_amount = min(maximum, min_raise + additional)
            # If the computed raise is not enough, fallback to call.
            if raise_amount <= call_amount:
                if call_amount > 0:
                    return PlayerAction.CALL, call_amount
                else:
                    return PlayerAction.CHECK, 0
            return PlayerAction.RAISE, raise_amount
        # Fallback
        return PlayerAction.FOLD, 0
