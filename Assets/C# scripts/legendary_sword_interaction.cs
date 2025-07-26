using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class legendary_sword_interaction : MonoBehaviour
{
    public string interactButtonName = "talk_text"; // Set this to the name of your interact button GameObject
    public float interactionRange = 1f; // Adjust this value based on your desired range

    private GameObject interactButton; // Reference to the UI interact button GameObject


    // Dialogue GameObject references
    private GameObject dialogue_one;
    private GameObject dialogue_two;
    private GameObject dialogue_three;
    


    // Start is called before the first frame update
    void Start()
    {
        dialogue_one = GameObject.Find("congrats_game_over_text");
        dialogue_two = GameObject.Find("legendary_sword_text");
        dialogue_three = GameObject.Find("thanks_for_playing_text");

        dialogue_one.SetActive(false);
        dialogue_two.SetActive(false);
        dialogue_three.SetActive(false);

        // Find the interact button GameObjectusing its name
        interactButton = GameObject.Find(interactButtonName);

        if (interactButton == null)
        {
            Debug.LogError("Interact button not found. Make sure to set the correct name.");
        }
    }


    // Update is called once per frame
    void Update()
    {
        CheckPlayerDistance();
    }


    void CheckPlayerDistance()
    {
        GameObject player = GameObject.FindGameObjectWithTag("Player");

        character_movement characterMovementVariable = player.GetComponent<character_movement>();
        character_jump_movement characterJumpMovementVariable = player.GetComponent<character_jump_movement>();


        //Trigger feature:
        if (player != null)
        {
            float distance = Vector2.Distance(transform.position, player.transform.position);

            if (distance <= interactionRange)
            {
                ShowInteractButton();


                // Dialogue features:
                // Allowing interaction if 
                if (Input.GetKeyDown(KeyCode.I))
                {


                    // Locking character movement feature:
                    if (characterMovementVariable != null)
                    {
                        // Halt the character's movement
                        characterMovementVariable.ToggleMovement(false);
                    }

                    // Locking character jump movement feature:
                    if (characterJumpMovementVariable != null)
                    {
                        // Halt the character's movement
                        characterJumpMovementVariable.ToggleJumpMovement(false);
                    }


                    if (!dialogue_one.activeSelf && !dialogue_two.activeSelf && !dialogue_three.activeSelf)
                    {
                        StartDialogue();
                    }
                    else if (dialogue_one.activeSelf && !dialogue_two.activeSelf && !dialogue_three.activeSelf)
                    {
                        ContinueDialogue();
                    }
                    else if (dialogue_one.activeSelf && dialogue_two.activeSelf && !dialogue_three.activeSelf)
                    {
                        EndDialogue();
                        characterMovementVariable.ToggleMovement(true);
                        characterJumpMovementVariable.ToggleJumpMovement(true);
                    }
                }


            }
            else
            {
                HideInteractButton();
            }
        }
    }

    void ShowInteractButton()
    {
        interactButton.SetActive(true);
    }

    void HideInteractButton()
    {

        interactButton.SetActive(false);

    }

    // Call this method for the first dialogue
    void StartDialogue()
    {
        dialogue_one.SetActive(true);
    }

    // Call this method when the first dialogue is complete
    void ContinueDialogue()
    {
        dialogue_two.SetActive(true);
    }

    // Call this method when the second dialogue is complete
    void EndDialogue()
    {
        dialogue_three.SetActive(true);
    }
}