using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class villagers_NPC_interaction : MonoBehaviour
{
    public string interactButtonName = "talk_text"; // Set this to the name of your interact button GameObject
    public float interactionRange = 1f; // Adjust this value based on your desired range

    private GameObject interactButton; // Reference to the UI interact button GameObject


    // Dialogue GameObject references
    public string textBubbleName = "text_bubble sprite_0";
    public string villager_dialogue_one_name = "dialogueOne_text";
    public string villager_dialogue_two_name = "dialogueTwo_text";
    private GameObject text_bubble;
    private GameObject villager_dialogue_one;
    private GameObject villager_dialogue_two;


    // Start is called before the first frame update
    void Start()
    {
        text_bubble = GameObject.Find(textBubbleName);
        villager_dialogue_one = GameObject.Find(villager_dialogue_one_name);
        villager_dialogue_two = GameObject.Find(villager_dialogue_two_name);

        text_bubble.SetActive(false);
        villager_dialogue_one.SetActive(false);
        villager_dialogue_two.SetActive(false);

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


                    // Locking character horizontal movement feature:
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


                    if (!text_bubble.activeSelf && !villager_dialogue_one.activeSelf && !villager_dialogue_two.activeSelf)
                    {
                        StartDialogue();
                    }
                    else if (text_bubble.activeSelf && villager_dialogue_one.activeSelf && !villager_dialogue_two.activeSelf)
                    {
                        ContinueDialogue();
                    }
                    else if (text_bubble.activeSelf && !villager_dialogue_one.activeSelf && villager_dialogue_two.activeSelf)
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
        text_bubble.SetActive(true);
        villager_dialogue_one.SetActive(true);
    }

    // Call this method when the first dialogue is complete
    void ContinueDialogue()
    {
        villager_dialogue_one.SetActive(false);
        villager_dialogue_two.SetActive(true);
    }

    // Call this method when the second dialogue is complete
    void EndDialogue()
    {
        villager_dialogue_two.SetActive(false);
        text_bubble.SetActive(false);
    }
}